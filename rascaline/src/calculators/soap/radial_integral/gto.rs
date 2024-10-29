use std::f64;

use ndarray::{Array2, ArrayViewMut1};

use crate::calculators::shared::basis::radial::GtoRadialBasis;
use crate::calculators::shared::{DensityKind, SoapRadialBasis};
use crate::math::{gamma, hyp1f1};
use crate::Error;

use super::SoapRadialIntegral;

/// Implementation of the radial integral for GTO radial basis and gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct SoapRadialIntegralGto {
    /// Which value of l/lambda is this radial integral for
    o3_lambda: usize,
    /// `σ^2`, with σ the atomic density gaussian width
    atomic_gaussian_width_2: f64,
    /// `1/2σ^2`, with σ the atomic density gaussian width
    atomic_gaussian_constant: f64,
    /// `1/2σ_n^2`, with `σ_n` the GTO gaussian width, i.e. `cutoff * max(√n, 1)
    /// / n_max`
    gto_gaussian_constants: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
}


impl SoapRadialIntegralGto {
    /// Create a new SOAP radial integral
    pub fn new(cutoff: f64, density: DensityKind, basis: &SoapRadialBasis, o3_lambda: usize) -> Result<SoapRadialIntegralGto, Error> {
        let gaussian_width = if let DensityKind::Gaussian { width } = density {
            width
        } else {
            return Err(Error::Internal("density must be Gaussian for the GTO radial integral".into()));
        };

        let &max_radial = if let SoapRadialBasis::Gto { max_radial, radius } = basis {
            if let Some(radius) = radius {
                #[allow(clippy::float_cmp)]
                if *radius != cutoff {
                    return Err(Error::Internal(
                        "GTO radius must be the same as the cutoff radius in SOAP, \
                        or should not provided".into()
                    ));
                }
            }
            max_radial
        } else {
            return Err(Error::Internal("radial basis must be GTO for the GTO radial integral".into()));
        };

        // these should be checked before we reach this function
        assert!(gaussian_width > 1e-16 && gaussian_width.is_finite());

        let basis = GtoRadialBasis {
            size: max_radial + 1,
            radius: cutoff,
        };
        let gto_gaussian_widths = basis.gaussian_widths();
        let gto_orthonormalization = basis.orthonormalization_matrix();

        let gto_gaussian_constants = gto_gaussian_widths.iter()
            .map(|&sigma| 1.0 / (2.0 * sigma * sigma))
            .collect::<Vec<_>>();

        let atomic_gaussian_width_2 = gaussian_width * gaussian_width;
        let atomic_gaussian_constant = 1.0 / (2.0 * atomic_gaussian_width_2);

        return Ok(SoapRadialIntegralGto {
            o3_lambda: o3_lambda,
            atomic_gaussian_width_2: atomic_gaussian_width_2,
            atomic_gaussian_constant: atomic_gaussian_constant,
            gto_gaussian_constants: gto_gaussian_constants,
            gto_orthonormalization: gto_orthonormalization.t().to_owned(),
        })
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



impl SoapRadialIntegral for SoapRadialIntegralGto {
    fn size(&self) -> usize {
        self.gto_gaussian_constants.len()
    }

    #[time_graph::instrument(name = "GtoRadialIntegral::compute")]
    fn compute(
        &self,
        distance: f64,
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

        // Define global factor of radial integral arising from three parts:
        // - a global 4 pi factor coming from integration of the angular part of
        //   the radial integral (see the docs for `SoapRadialIntegral`)
        // - a global factor of sqrt(pi)/4 from the calculation of the integral
        //   of GTO basis functions and gaussian density
        // - the normalization constant of the atomic Gaussian density. We use a
        //   factor of 1/(pi*sigma^2)^0.75 which leads to Gaussian densities
        //   that are normalized in the L2-sense, i.e. integral_{R^3} |g(r)|^2
        //   d^3r = 1.
        //
        // These three factors simplify to (pi/sigma^2)^3/4
        let global_factor = (std::f64::consts::PI / self.atomic_gaussian_width_2).powf(0.75);

        let c = self.atomic_gaussian_constant;
        let c_rij = c * distance;
        let c_rij_l = c_rij.powi(self.o3_lambda as i32);
        let exp_c_rij = f64::exp(-distance * c_rij);

        // `global_factor * exp(-c rij^2) * (c * rij)^l`
        let factor = global_factor * exp_c_rij * c_rij_l;

        for n in 0..self.size() {
            let gto_constant = self.gto_gaussian_constants[n];

            let z = c_rij * c_rij / (c + gto_constant);
            // Calculate Gamma(a) / Gamma(b) 1F1(a, b, z)
            double_regularized_1f1(self.o3_lambda, n, z, &mut values[n], gradients.as_mut().map(|g| &mut g[n]));
            if !values[n].is_finite() {
                    panic!(
                        "Failed to compute radial integral with GTO basis. \
                        Try increasing decreasing the `cutoff`, or increasing \
                        the Gaussian's `width`."
                    );
                }

            let n_l_3_over_2 = 0.5 * (n + self.o3_lambda) as f64 + 1.5;
            let c_dn = (c + gto_constant).powf(-n_l_3_over_2);

            values[n] *= c_dn * factor;
            if let Some(ref mut gradients) = gradients {
                gradients[n] *= c_dn * factor * 2.0 * z / distance;
                gradients[n] += values[n] * (self.o3_lambda as f64 / distance - 2.0 * c_rij);
            }
        }

        // for r = 0, the formula used in the calculations above yield NaN,
        // which in turns breaks the SplinedGto radial integral. From the
        // analytical formula, the gradient is 0 everywhere expect for l=1
        if distance == 0.0 {
            if let Some(ref mut gradients) = gradients {
                if self.o3_lambda == 1 {
                    for n in 0..self.size() {
                        let gto_constant = self.gto_gaussian_constants[n];
                        let a = 0.5 * (n + self.o3_lambda) as f64 + 1.5;
                        let b = 2.5;
                        let c_dn = (c + gto_constant).powf(-a);
                        let factor = global_factor * c * c_dn;

                        gradients[n] = gamma(a) / gamma(b) * factor;
                    }
                } else {
                    gradients.fill(0.0);
                }
            }
        }

        values.assign(&values.dot(&self.gto_orthonormalization));
        if let Some(ref mut gradients) = gradients {
            gradients.assign(&gradients.dot(&self.gto_orthonormalization));
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use super::super::SoapRadialIntegral;
    use ndarray::Array1;

    #[test]
    #[should_panic = "radial overlap matrix is singular, try with a lower max_radial (current value is 30)"]
    fn ill_conditioned_orthonormalization() {
        let density = DensityKind::Gaussian { width: 0.4 };
        let basis = SoapRadialBasis::Gto { max_radial: 30, radius: None };
        SoapRadialIntegralGto::new(5.0, density, &basis, 0).unwrap();
    }

    #[test]
    #[should_panic = "wrong size for values array, expected [4] but got [3]"]
    fn values_array_size() {
        let density = DensityKind::Gaussian { width: 0.4 };
        let basis = SoapRadialBasis::Gto { max_radial: 3, radius: None };
        let gto = SoapRadialIntegralGto::new(5.0, density, &basis, 0).unwrap();
        let mut values = Array1::from_elem(3, 0.0);

        gto.compute(1.0, values.view_mut(), None);
    }

    #[test]
    #[should_panic = "wrong size for gradients array, expected [4] but got [3]"]
    fn gradient_array_size() {
        let density = DensityKind::Gaussian { width: 0.5 };
        let basis = SoapRadialBasis::Gto { max_radial: 3, radius: None };
        let gto = SoapRadialIntegralGto::new(5.0, density, &basis, 0).unwrap();

        let mut values = Array1::from_elem(4, 0.0);
        let mut gradients = Array1::from_elem(3, 0.0);

        gto.compute(1.0, values.view_mut(), Some(gradients.view_mut()));
    }

    #[test]
    fn gradients_near_zero() {
        let density = DensityKind::Gaussian { width: 0.5 };
        let basis = SoapRadialBasis::Gto { max_radial: 7, radius: None };

        for l in 0..4 {
            let gto_ri = SoapRadialIntegralGto::new(3.4, density, &basis, l).unwrap();

            let mut values = Array1::from_elem(8, 0.0);
            let mut gradients = Array1::from_elem(8, 0.0);
            let mut gradients_plus = Array1::from_elem(8, 0.0);
            gto_ri.compute(0.0, values.view_mut(), Some(gradients.view_mut()));
            gto_ri.compute(1e-12, values.view_mut(), Some(gradients_plus.view_mut()));

            assert_relative_eq!(
                gradients, gradients_plus, epsilon=1e-11, max_relative=1e-6,
            );
        }
    }

    #[test]
    fn finite_differences() {
        let density = DensityKind::Gaussian { width: 0.5 };
        let basis = SoapRadialBasis::Gto { max_radial: 7, radius: None };

        let x = 3.4;
        let delta = 1e-9;

        for l in 0..=8 {
            let gto_ri = SoapRadialIntegralGto::new(5.0, density, &basis, l).unwrap();

            let mut values = Array1::from_elem(8, 0.0);
            let mut values_delta = Array1::from_elem(8, 0.0);
            let mut gradients = Array1::from_elem(8, 0.0);
            gto_ri.compute(x, values.view_mut(), Some(gradients.view_mut()));
            gto_ri.compute(x + delta, values_delta.view_mut(), None);

            let finite_differences = (&values_delta - &values) / delta;

            assert_relative_eq!(
                finite_differences, gradients, max_relative=1e-4
            );
        }
    }
}
