use std::f64;

use ndarray::{Array2, Array1, ArrayViewMut1};

use crate::calculators::shared::basis::radial::GtoRadialBasis;
use crate::calculators::shared::{DensityKind, LodeRadialBasis};
use crate::math::{hyp2f1, hyp1f1, gamma};
use crate::Error;

use super::LodeRadialIntegral;

/// Implementation of the LODE radial integral for GTO radial basis and Gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct LodeRadialIntegralGto {
    /// Which value of l/lambda is this radial integral for
    o3_lambda: usize,
    /// `1/2σ_n^2`, with `σ_n` the GTO gaussian width, i.e. `cutoff * max(√n, 1)
    /// / n_max`
    gto_gaussian_widths: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
}

impl LodeRadialIntegralGto {
    /// Create a new LODE radial integral
    pub fn new(basis: &LodeRadialBasis, o3_lambda: usize) -> Result<LodeRadialIntegralGto, Error> {
        let (&max_radial, &gto_radius) = if let LodeRadialBasis::Gto { max_radial, radius } = basis {
            (max_radial, radius)
        } else {
            return Err(Error::Internal("radial basis must be GTO for the GTO radial integral".into()));
        };

        if gto_radius < 1e-16 {
            return Err(Error::InvalidParameter(
                "radius of GTO radial basis can not be negative".into()
            ));
        } else if !gto_radius.is_finite() {
            return Err(Error::InvalidParameter(
                "radius of GTO radial basis can not be infinite/NaN".into()
            ));
        }

        let basis = GtoRadialBasis {
            size: max_radial + 1,
            radius: gto_radius,
        };
        let gto_gaussian_widths = basis.gaussian_widths();
        let gto_orthonormalization = basis.orthonormalization_matrix();

        return Ok(LodeRadialIntegralGto {
            o3_lambda: o3_lambda,
            gto_gaussian_widths: gto_gaussian_widths,
            gto_orthonormalization: gto_orthonormalization.t().to_owned(),
        })
    }
}

impl LodeRadialIntegral for LodeRadialIntegralGto {
    fn size(&self) -> usize {
        self.gto_gaussian_widths.len()
    }

    #[time_graph::instrument(name = "LodeRadialIntegralGto::compute")]
    fn compute(
        &self,
        k_norm: f64,
        mut values: ArrayViewMut1<f64>,
        mut gradients: Option<ArrayViewMut1<f64>>
    ) {
        assert_eq!(
            values.shape(), [self.size()],
            "wrong size for values array, expected [{}] but got [{}]",
            self.size(), values.shape()[0]
        );

        if let Some(ref gradients) = gradients {
            assert_eq!(
                gradients.shape(), [self.size()],
                "wrong size for gradients array, expected [{}] but got [{}]",
                self.size(), gradients.shape()[0]
            );
        }

        let global_factor = std::f64::consts::PI.sqrt() / std::f64::consts::SQRT_2;

        for n in 0..self.size() {
            let sigma_n = self.gto_gaussian_widths[n];
            let k_sigma_n_sqrt2 = k_norm * sigma_n / std::f64::consts::SQRT_2;
            // `global_factor * sqrt(2)^{n} * sigma_n^{n + 3} * (k * sigma_n / sqrt(2))^l`
            let factor = global_factor
                * sigma_n.powi(n as i32 + 3) * std::f64::consts::SQRT_2.powi(n as i32)
                * k_sigma_n_sqrt2.powi(self.o3_lambda as i32);

            let k_norm_sigma_n_2 = - k_norm * sigma_n * sigma_n;
            let z = 0.5 * k_norm * k_norm_sigma_n_2;

            double_regularized_1f1(
                self.o3_lambda, n, z, &mut values[n], gradients.as_mut().map(|g| &mut g[n])
            );

            assert!(values[n].is_finite());
            values[n] *= factor;
            if let Some(ref mut gradients) = gradients {
                gradients[n] *= k_norm_sigma_n_2 * factor;
                gradients[n] += self.o3_lambda as f64 / k_norm * values[n];
            }
        }

        // for k_norm = 0, the formula used in the calculations above yield NaN,
        // which in turns breaks the SplinedGto radial integral. From the
        // analytical formula, the gradient is 0 everywhere expect for l=1
        if k_norm == 0.0 {
            if let Some(ref mut gradients) = gradients {
                gradients.fill(0.0);

                if self.o3_lambda == 1 {
                    for n in 0..self.size() {
                        let sigma_n = self.gto_gaussian_widths[n];
                        let a = 0.5 * (n + self.o3_lambda) as f64 + 1.5;
                        let b = 2.5;
                        let factor = global_factor * sigma_n.powi((n + self.o3_lambda) as i32 + 3) * std::f64::consts::SQRT_2.powi(n as i32 - self.o3_lambda as i32);

                        gradients[n] = gamma(a) / gamma(b) * factor;
                    }
                }
            }
        }

        values.assign(&values.dot(&self.gto_orthonormalization));
        if let Some(ref mut gradients) = gradients {
            gradients.assign(&gradients.dot(&self.gto_orthonormalization));
        }
    }

    fn get_center_contribution(&self, density: DensityKind) -> Result<Array1<f64>, Error> {
        let radial_size = self.gto_gaussian_widths.len();

        let (smearing, exponent) = match density {
            DensityKind::SmearedPowerLaw { smearing, exponent } => {
                (smearing, exponent as f64)
            }
            _ => {
                return Err(Error::InvalidParameter(
                    "Only 'SmearedPowerLaw' density is supported in LODE".into()
                ));
            }
        };

        let mut contrib = Array1::from_elem(radial_size, 0.0);


        let n_eff: Vec<f64> = (0..radial_size)
            .map(|n| 0.5 * (3.0 + n as f64))
            .collect();

        if exponent == 0.0 {
            let factor = std::f64::consts::PI.powf(-0.25) / (smearing * smearing).powf(0.75);

            for n in 0..radial_size {
                let alpha = 0.5 * (1.0 / (smearing * smearing)
                    + 1.0 / (self.gto_gaussian_widths[n] * self.gto_gaussian_widths[n]));
                contrib[n] = factor * gamma(n_eff[n]) / alpha.powf(n_eff[n]);
            }
        } else {
            let factor = 2.0 * f64::sqrt(4.0 * std::f64::consts::PI)
                / gamma(exponent / 2.0)
                / exponent;

            for n in 0..radial_size {
                let s = smearing / self.gto_gaussian_widths[n];
                let hyparg = 1.0 / (1.0 + s * s);

                contrib[n] = factor
                    * f64::powf(2.0, (1.0 + n as f64 - exponent) / 2.0)
                    * smearing.powi(3 + n as i32 - exponent as i32)
                    * gamma(n_eff[n])
                    * hyp2f1(1.0, n_eff[n], (exponent + 2.0) / 2.0, hyparg)
                    * hyparg.powf(n_eff[n]);
            }
        }

        return Ok(contrib.dot(&self.gto_orthonormalization));
    }
}

#[inline]
fn hyp1f1_derivative(a: f64, b: f64, x: f64) -> f64 {
    a / b * hyp1f1(a + 1.0, b + 1.0, x)
}

#[inline]
#[allow(clippy::many_single_char_names)]
/// Compute `G(a, b, z) = Gamma(a) / Gamma(b) 1F1(a, b, z)` for
/// `a = 1/2 (n + l + 3)` and `b = l + 3/2`.
///
/// This is similar (but not the exact same) to the G function defined in
/// appendix A in <https://doi.org/10.1063/5.0044689>.
///
/// The function is called "double regularized 1F1" by reference to the
/// "regularized 1F1" function (i.e. `1F1(a, b, z) / Gamma(b)`)
fn double_regularized_1f1(l: usize, n: usize, z: f64, value: &mut f64, gradient: Option<&mut f64>) {
    let (a, b) = (0.5 * (n + l + 3) as f64, l as f64 + 1.5);
    let ratio = gamma(a) / gamma(b);

    *value = ratio * hyp1f1(a, b, z);
    if let Some(gradient) = gradient {
        *gradient = ratio * hyp1f1_derivative(a, b, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn gradients_near_zero() {
        let radial_size = 8;
        for o3_lambda in [0, 1, 3, 5, 8] {
            let gto = LodeRadialIntegralGto::new(
                &LodeRadialBasis::Gto { max_radial: (radial_size - 1), radius: 5.0 },
                o3_lambda
            ).unwrap();

            let mut values = Array1::from_elem(radial_size, 0.0);
            let mut gradients = Array1::from_elem(radial_size, 0.0);
            let mut gradients_plus = Array1::from_elem(radial_size, 0.0);
            gto.compute(0.0, values.view_mut(), Some(gradients.view_mut()));
            gto.compute(1e-12, values.view_mut(), Some(gradients_plus.view_mut()));

            assert_relative_eq!(
                gradients, gradients_plus, epsilon=1e-9, max_relative=1e-6,
            );
        }
    }

    #[test]
    fn finite_differences() {
        let k = 3.4;
        let delta = 1e-6;

        let radial_size = 8;

        for o3_lambda in [0, 1, 3, 5, 8] {
            let gto = LodeRadialIntegralGto::new(
                &LodeRadialBasis::Gto { max_radial: (radial_size - 1), radius: 5.0 },
                o3_lambda
            ).unwrap();

            let mut values = Array1::from_elem(radial_size, 0.0);
            let mut values_delta = Array1::from_elem(radial_size, 0.0);
            let mut gradients = Array1::from_elem(radial_size, 0.0);
            gto.compute(k, values.view_mut(), Some(gradients.view_mut()));
            gto.compute(k + delta, values_delta.view_mut(), None);

            let finite_differences = (&values_delta - &values) / delta;

            assert_relative_eq!(
                finite_differences, gradients, max_relative=1e-4
            );
        }
    }

    #[test]
    fn central_atom_contribution() {
        let potential_exponents = [0, 1, 2, 6];

        // Reference values taken from pyLODE
        let reference_vals = [
            [7.09990773e-01, 6.13767550e-01, 3.34161655e-01, 8.35301652e-02, 1.78439072e-02, -3.44944648e-05],
            [1.69193719, 2.02389574, 2.85086136, 3.84013091, 1.62869125, 7.03338899],
            [1.00532822, 1.10024472, 1.34843326, 1.19816598, 0.69150744, 1.2765415],
            [0.03811939, 0.03741200, 0.03115835, 0.01364843, 0.00534184, 0.00205973]];

        for (i, &p) in potential_exponents.iter().enumerate() {
            let gto = LodeRadialIntegralGto::new(
                &LodeRadialBasis::Gto { max_radial: 5, radius: 5.0 }, 0
            ).unwrap();

            let density = DensityKind::SmearedPowerLaw { smearing: 1.0, exponent: p };

            let center_contrib = gto.get_center_contribution(density).unwrap();
            assert_relative_eq!(center_contrib, ndarray::arr1(&reference_vals[i]), max_relative=3e-6);
        };
    }
}
