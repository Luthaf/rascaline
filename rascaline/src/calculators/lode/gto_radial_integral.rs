use std::f64;

use ndarray::{Array2, ArrayViewMut2};

use crate::math::{gamma, DoubleRegularized1F1};
use crate::Error;

use crate::calculators::radial_integral::RadialIntegral;
use crate::calculators::radial_integral::GtoRadialBasis;

pub use super::super::soap::GtoParameters;

/// Implementation of the LODE radial integral for GTO radial basis and Gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct LodeGtoRadialIntegral {
    parameters: GtoParameters,
    /// σ_n GTO gaussian width, i.e. `cutoff * max(√n, 1) / n_max`
    gto_gaussian_widths: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
    /// Implementation of `Gamma(a) / Gamma(b) 1F1(a, b, z)`
    double_regularized_1f1: DoubleRegularized1F1,
}

impl LodeGtoRadialIntegral {
    pub fn new(parameters: GtoParameters) -> Result<LodeGtoRadialIntegral, Error> {
        parameters.validate()?;

        let gto_gaussian_widths = GtoRadialBasis::gaussian_widths(parameters.max_radial, parameters.cutoff);

        let gto_orthonormalization = GtoRadialBasis::orthonormalization_matrix(
            parameters.max_radial, parameters.cutoff
        );

        return Ok(LodeGtoRadialIntegral {
            parameters: parameters,
            double_regularized_1f1: DoubleRegularized1F1 {
                max_angular: parameters.max_angular,
            },
            gto_gaussian_widths: gto_gaussian_widths,
            gto_orthonormalization: gto_orthonormalization,
        })
    }
}

impl RadialIntegral for LodeGtoRadialIntegral {
    #[time_graph::instrument(name = "LodeGtoRadialIntegral::compute")]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculators::radial_integral::RadialIntegral;

    use ndarray::Array2;
    use approx::assert_relative_eq;

    #[test]
    fn gradients_near_zero() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = LodeGtoRadialIntegral::new(GtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
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
        let gto = LodeGtoRadialIntegral::new(GtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
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
}
