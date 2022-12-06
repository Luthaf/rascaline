use std::f64;

use ndarray::{Array1, Array2, ArrayViewMut2};

use crate::math::{hyp2f1, gamma, DoubleRegularized1F1};
use crate::Error;

use super::LodeRadialIntegral;
use crate::calculators::radial_basis::GtoRadialBasis;

/// Parameters controlling the LODE radial integral with GTO radial basis
#[derive(Debug, Clone, Copy)]
pub struct LodeRadialIntegralGtoParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// atomic density gaussian width
    pub atomic_gaussian_width: f64,
    /// potential exponent
    pub potential_exponent: usize,
    /// cutoff radius
    pub cutoff: f64,
}

impl LodeRadialIntegralGtoParameters {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.max_radial == 0 {
            return Err(Error::InvalidParameter(
                "max_radial must be at least 1 for GTO radial integral".into()
            ));
        }

        if self.cutoff < 0.0 || !self.cutoff.is_finite() {
            return Err(Error::InvalidParameter(
                "cutoff must be a positive number for GTO radial integral".into()
            ));
        }

        if self.atomic_gaussian_width < 0.0 || !self.atomic_gaussian_width.is_finite() {
            return Err(Error::InvalidParameter(
                "atomic_gaussian_width must be a positive number for GTO radial integral".into()
            ));
        }

        Ok(())
    }
}

/// Implementation of the LODE radial integral for GTO radial basis and Gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct LodeRadialIntegralGto {
    parameters: LodeRadialIntegralGtoParameters,
    /// σ_n GTO gaussian width, i.e. `cutoff * max(√n, 1) / n_max`
    gto_gaussian_widths: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
    /// Implementation of `Gamma(a) / Gamma(b) 1F1(a, b, z)`
    double_regularized_1f1: DoubleRegularized1F1,
}

impl LodeRadialIntegralGto {
    pub fn new(parameters: LodeRadialIntegralGtoParameters) -> Result<LodeRadialIntegralGto, Error> {
        parameters.validate()?;

        let basis = GtoRadialBasis {
            max_radial: parameters.max_radial,
            cutoff: parameters.cutoff,
        };
        let gto_gaussian_widths = basis.gaussian_widths();
        let gto_orthonormalization = basis.orthonormalization_matrix();

        return Ok(LodeRadialIntegralGto {
            parameters: parameters,
            double_regularized_1f1: DoubleRegularized1F1 {
                max_angular: parameters.max_angular,
            },
            gto_gaussian_widths: gto_gaussian_widths,
            gto_orthonormalization: gto_orthonormalization.t().to_owned(),
        })
    }
}

impl LodeRadialIntegral for LodeRadialIntegralGto {
    #[time_graph::instrument(name = "LodeRadialIntegralGto::compute")]
    fn compute(
        &self,
        k_norm: f64,
        mut values: ArrayViewMut2<f64>,
        mut gradients: Option<ArrayViewMut2<f64>>
    ) {
        let expected_shape = [self.parameters.max_angular + 1, self.parameters.max_radial];
        assert_eq!(
            values.shape(), expected_shape,
            "wrong size for values array, expected [{}, {}] but got [{}, {}]",
            expected_shape[0], expected_shape[1], values.shape()[0], values.shape()[1]
        );

        if let Some(ref gradients) = gradients {
            assert_eq!(
                gradients.shape(), expected_shape,
                "wrong size for gradients array, expected [{}, {}] but got [{}, {}]",
                expected_shape[0], expected_shape[1], gradients.shape()[0], gradients.shape()[1]
            );
        }

        let global_factor = std::f64::consts::PI.sqrt() / std::f64::consts::SQRT_2;

        for n in 0..self.parameters.max_radial {
            let sigma_n = self.gto_gaussian_widths[n];
            let k_sigma_n_sqrt2 = k_norm * sigma_n / std::f64::consts::SQRT_2;
            // `global_factor * sqrt(2)^{n} * sigma_n^{n + 3} * (k * sigma_n / sqrt(2))^l`
            let mut factor = global_factor * sigma_n.powi(n as i32 + 3) * std::f64::consts::SQRT_2.powi(n as i32);

            let k_norm_sigma_n_2 = - k_norm * sigma_n * sigma_n;
            let z = 0.5 * k_norm * k_norm_sigma_n_2;
            self.double_regularized_1f1.compute(
                z, n,
                values.index_axis_mut(ndarray::Axis(1), n),
                gradients.as_mut().map(|g| g.index_axis_mut(ndarray::Axis(1), n))
            );

            for l in 0..(self.parameters.max_angular + 1) {
                values[[l, n]] *= factor;
                if let Some(ref mut gradients) = gradients {
                    gradients[[l, n]] *= k_norm_sigma_n_2 * factor;
                    gradients[[l, n]] += l as f64 / k_norm * values[[l, n]];
                }

                factor *= k_sigma_n_sqrt2;
            }
        }

        // for k_norm = 0, the formula used in the calculations above yield NaN,
        // which in turns breaks the SplinedGto radial integral. From the
        // analytical formula, the gradient is 0 everywhere expect for l=1
        if k_norm == 0.0 {
            if let Some(ref mut gradients) = gradients {
                gradients.fill(0.0);

                if self.parameters.max_angular >= 1 {
                    let l = 1;
                    for n in 0..self.parameters.max_radial {
                        let sigma_n = self.gto_gaussian_widths[n];
                        let a = 0.5 * (n + l) as f64 + 1.5;
                        let b = 2.5;
                        let factor = global_factor * sigma_n.powi((n + l) as i32 + 3) * std::f64::consts::SQRT_2.powi(n as i32 - l as i32);

                        gradients[[l, n]] = gamma(a) / gamma(b) * factor;
                    }
                }
            }
        }

        values.assign(&values.dot(&self.gto_orthonormalization));
        if let Some(ref mut gradients) = gradients {
            gradients.assign(&gradients.dot(&self.gto_orthonormalization));
        }
    }

    fn compute_center_contribution(&self) -> Array1<f64> {
        let max_radial = self.parameters.max_radial;
        let atomic_gaussian_width = self.parameters.atomic_gaussian_width;
        let potential_exponent = self.parameters.potential_exponent as f64;

        let mut contrib = Array1::from_elem(max_radial, 0.0);

        let basis = GtoRadialBasis {
            max_radial,
            cutoff: self.parameters.cutoff,
        };
        let gto_gaussian_widths = basis.gaussian_widths();
        let n_eff: Vec<f64> = (0..max_radial)
            .into_iter()
            .map(|n| 0.5 * (3. + n as f64))
            .collect();

        if potential_exponent == 0. {
            let factor = std::f64::consts::PI.powf(-0.25)
                / (atomic_gaussian_width * atomic_gaussian_width).powf(0.75);

            for n in 0..max_radial {
                let alpha = 0.5
                    * (1. / (atomic_gaussian_width * atomic_gaussian_width)
                        + 1. / (gto_gaussian_widths[n] * gto_gaussian_widths[n]));
                contrib[n] = factor * gamma(n_eff[n]) / alpha.powf(n_eff[n]);
            }
        } else {
            let factor = 2. * f64::sqrt(4. * std::f64::consts::PI)
                / gamma(potential_exponent / 2.)
                / potential_exponent;

            for n in 0..max_radial {
                let s = atomic_gaussian_width / gto_gaussian_widths[n];
                let hyparg = 1. / (1. + s * s);

                contrib[n] = factor
                    * 2_f64.powf((1. + n as f64 - potential_exponent) / 2.)
                    * atomic_gaussian_width.powi(3 + n as i32 - potential_exponent as i32)
                    * gamma(n_eff[n])
                    * hyp2f1(1., n_eff[n], (potential_exponent + 2.) / 2., hyparg)
                    * hyparg.powf(n_eff[n]);
            }
        }

        let gto_orthonormalization = basis.orthonormalization_matrix();

        return gto_orthonormalization.dot(&(contrib));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array2;
    use approx::assert_relative_eq;

    #[test]
    fn gradients_near_zero() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        let mut gradients_plus = Array2::from_elem(shape, 0.0);
        gto.compute(0.0, values.view_mut(), Some(gradients.view_mut()));
        gto.compute(1e-12, values.view_mut(), Some(gradients_plus.view_mut()));

        assert_relative_eq!(
            gradients, gradients_plus, epsilon=1e-9, max_relative=1e-6,
        );
    }

    #[test]
    fn finite_differences() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        let k = 3.4;
        let delta = 1e-6;

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut values_delta = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        gto.compute(k, values.view_mut(), Some(gradients.view_mut()));
        gto.compute(k + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;

        assert_relative_eq!(
            finite_differences, gradients, max_relative=1e-4
        );
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

        for (i, &p) in potential_exponents.iter().enumerate(){
            let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
                cutoff: 5.0,
                max_radial: 6,
                max_angular: 2,
                atomic_gaussian_width: 1.0,
                potential_exponent: p,
            }).unwrap();

            let center_contrib = gto.compute_center_contribution();
            assert_relative_eq!(center_contrib, ndarray::arr1(&reference_vals[i]), max_relative=3e-6);
        };
    }
}
