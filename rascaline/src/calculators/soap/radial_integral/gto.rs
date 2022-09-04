use std::f64;

use ndarray::{Array2, ArrayViewMut2, Array1};

use crate::math::{gamma, hyp1f1, hyp1f1_derivative};
use crate::Error;

use super::RadialIntegral;

const SQRT_PI_OVER_4: f64 = 0.44311346272637897;

/// Parameters controlling GTO radial basis
#[derive(Debug, Clone, Copy)]
pub struct GtoParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// atomic density gaussian width
    pub atomic_gaussian_width: f64,
    /// cutoff radius
    pub cutoff: f64,
}

impl GtoParameters {
    fn validate(&self) -> Result<(), Error> {
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

/// Compute the G function from the [2021 paper], appendix C:
///
/// `G(rij) = Γ(a) / Γ(b) exp(-c rij^2) 1F1(a, b, c^2 rij^2 / (c + d))`
///
/// [2021 paper]: https://doi.org/10.1063/5.0044689
#[derive(Debug, Clone)]
struct SphericalExpansionHyperGeometric {
    /// Γ(a) / Γ(b)
    gamma_ratio: f64,
    /// c in the expression above
    atomic_gaussian_constant: f64,
    /// d in the expression above
    gto_gaussian_constant: f64,
    /// `a` parameter of the function
    a: f64,
    /// `b` parameter of the function
    b: f64,
}

impl SphericalExpansionHyperGeometric {
    fn new(
        a: f64,
        b: f64,
        atomic_gaussian_constant: f64,
        gto_gaussian_constant: f64,
    ) -> SphericalExpansionHyperGeometric {
        SphericalExpansionHyperGeometric {
            gamma_ratio: gamma(a) / gamma(b),
            atomic_gaussian_constant,
            gto_gaussian_constant,
            a,
            b,
        }
    }

    fn compute(&self, rij: f64, gradients: bool) -> (f64, f64) {
        let alpha_rij = self.atomic_gaussian_constant * rij;

        let z = alpha_rij * alpha_rij / (self.atomic_gaussian_constant + self.gto_gaussian_constant);

        let exp_z = f64::exp(-rij * alpha_rij);
        let hyp1f1_z = hyp1f1(self.a, self.b, z);
        let value = self.gamma_ratio * exp_z * hyp1f1_z;

        let mut gradient = 0.0;
        if gradients {
            gradient = z / rij * hyp1f1_derivative(self.a, self.b, z) - alpha_rij * hyp1f1_z;
            gradient *= 2.0 * self.gamma_ratio * exp_z;
        }

        return (value, gradient);
    }
}

/// Implementation of the radial integral for GTO radial basis and gaussian
/// atomic density.
#[derive(Debug, Clone)]
pub struct GtoRadialIntegral {
    parameters: GtoParameters,

    /// 1/2σ^2, with σ the atomic density gaussian width
    atomic_gaussian_constant: f64,
    /// 1/2σ_n^2, with σ_n the GTO gaussian width, i.e. `cutoff * max(√n, 1) / n_max`
    gto_gaussian_constants: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO
    gto_orthonormalization: Array2<f64>,
    /// (l_max + 1) x (n_max) array of functions computing
    ///
    /// `G(rij) = Γ(a) / Γ(b) exp(-c rij^2) 1F1(a, b, c^2 rij^2 / (c + d))`
    ///
    /// where `a = (n + l + 3) / 2`, `b = l + 3 / 2`, `c = 1 / 2σ^2`, `d = 1 / 2
    /// (r_cut \sqrt(n) / (n_max + 1))^2`; σ is the atomic density gaussian
    /// width, `r_cut` the cutoff radius, `n_max` the number of radial basis, n
    /// the current radial basis index and l the current angular index.
    g_functions: Array2<SphericalExpansionHyperGeometric>,
}

fn gto_overlap_matrix(max_radial: usize, gto_gaussian_widths: &[f64]) -> Array2<f64> {
    let mut overlap = Array2::from_elem((max_radial, max_radial), 0.0);
    for n1 in 0..max_radial {
        let sigma1 = gto_gaussian_widths[n1];
        let sigma1_sq = sigma1 * sigma1;
        for n2 in n1..max_radial {
            let sigma2 = gto_gaussian_widths[n2];
            let sigma2_sq = sigma2 * sigma2;

            let n1_n2_3_over_2 = 0.5 * (3.0 + n1 as f64 + n2 as f64);
            let value =
                (0.5 / sigma1_sq + 0.5 / sigma2_sq).powf(-n1_n2_3_over_2)
                / (sigma1.powi(n1 as i32) * sigma2.powi(n2 as i32))
                * gamma(n1_n2_3_over_2)
                / ((sigma1 * sigma2).powf(1.5) * f64::sqrt(gamma(n1 as f64 + 1.5) * gamma(n2 as f64 + 1.5)));


            overlap[(n2, n1)] = value;
            overlap[(n1, n2)] = value;
        }
    }
    return overlap;
}

impl GtoRadialIntegral {
    pub fn new(parameters: GtoParameters) -> Result<GtoRadialIntegral, Error> {
        parameters.validate()?;

        let gto_gaussian_widths = (0..parameters.max_radial).into_iter().map(|n| {
            let n = n as f64;
            let n_max = parameters.max_radial as f64;
            parameters.cutoff * f64::max(f64::sqrt(n), 1.0) / n_max
        }).collect::<Vec<_>>();

        let gto_gaussian_constants = gto_gaussian_widths.iter()
            .map(|&sigma| 1.0 / (2.0 * sigma * sigma))
            .collect::<Vec<_>>();

        let gto_normalization = gto_gaussian_widths.iter()
            .zip(0..parameters.max_radial)
            .map(|(sigma, n)| f64::sqrt(2.0 / (sigma.powi(2 * n as i32 + 3) * gamma(n as f64 + 1.5))))
            .collect::<Array1<_>>();

        let overlap = gto_overlap_matrix(parameters.max_radial, &gto_gaussian_widths);
        // compute overlap^-1/2 through its eigendecomposition
        let mut eigen = crate::math::SymmetricEigen::new(overlap);
        for n in 0..parameters.max_radial {
            if eigen.eigenvalues[n] <= f64::EPSILON {
                panic!(
                    "radial overlap matrix is singular, try with a lower \
                    max_radial (current value is {})", parameters.max_radial
                );
            }
            eigen.eigenvalues[n] = 1.0 / f64::sqrt(eigen.eigenvalues[n]);
        }
        let gto_orthonormalization = eigen.recompose().dot(&Array2::from_diag(&gto_normalization));

        let sigma2 = parameters.atomic_gaussian_width * parameters.atomic_gaussian_width;
        let atomic_gaussian_constant = 1.0 / (2.0 * sigma2);

        let mut g_functions = Vec::new();
        g_functions.reserve(parameters.max_angular * (parameters.max_radial + 1));

        for l in 0..(parameters.max_angular + 1) {
            for n in 0..parameters.max_radial {
                let a = 0.5 * (n + l + 3) as f64;
                let b = l as f64 + 1.5;
                g_functions.push(SphericalExpansionHyperGeometric::new(
                    a, b, atomic_gaussian_constant, gto_gaussian_constants[n]
                ));
            }
        }

        let g_functions = Array2::from_shape_vec(
            (parameters.max_angular + 1, parameters.max_radial), g_functions
        ).expect("wrong shape in Array2::from_shape_vec");

        return Ok(GtoRadialIntegral {
            parameters: parameters,
            g_functions: g_functions,
            atomic_gaussian_constant: atomic_gaussian_constant,
            gto_gaussian_constants: gto_gaussian_constants,
            gto_orthonormalization: gto_orthonormalization.t().to_owned(),
        })
    }
}

impl RadialIntegral for GtoRadialIntegral {
    #[time_graph::instrument(name = "GtoRadialIntegral::compute")]
    fn compute(
        &self,
        distance: f64,
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

        let do_gradients = gradients.is_some();
        for l in 0..=self.parameters.max_angular {
            for n in 0..self.parameters.max_radial {
                let (g, g_grad) = self.g_functions[[l, n]].compute(distance, do_gradients);
                values[[l, n]] = g;
                if let Some(ref mut gradients) = gradients {
                    gradients[[l, n]] = g_grad;
                }
            }
        }

        // Define global factor of radial integral arising from two parts:
        // - a global factor of sqrt(pi)/4 from the integral itself
        // - the normalization constant of the atomic Gaussian density.
        //   We use a factor of 1/(pi*sigma^2)^0.75 which leads to
        //   Gaussian densities that are normalized in the L2-sense, i.e.
        //   integral_{R^3} |g(r)|^2 d^3r = 1.
        let atomic_sigma_sq = 0.5 / self.atomic_gaussian_constant;
        let atomic_gaussian_normalization = (std::f64::consts::PI * atomic_sigma_sq).powf(-0.75);
        let global_factor = SQRT_PI_OVER_4 * atomic_gaussian_normalization;

        let c = self.atomic_gaussian_constant;
        let c_rij = c * distance;

        for n in 0..self.parameters.max_radial {
            let gto_constant = self.gto_gaussian_constants[n];
            // `(c * rij)^l`
            let mut c_rij_l = 1.0;

            for l in 0..(self.parameters.max_angular + 1) {
                let n_l_3_over_2 = 0.5 * (n + l) as f64 + 1.5;
                let c_dn = (c + gto_constant).powf(-n_l_3_over_2);
                let factor = global_factor * c_rij_l * c_dn;
                c_rij_l *= c_rij;

                values[[l, n]] *= factor;
                if let Some(ref mut gradients) = gradients {
                    gradients[[l, n]] *= factor;
                    gradients[[l, n]] += values[[l, n]] * l as f64 / distance;
                }
            }
        }

        // for r = 0, the formula used in the calculations above yield NaN,
        // which in turns breaks the SplinedGto radial integral. From the
        // analytical formula, the gradient is 0 everywhere expect for l=1
        if distance == 0.0 {
            if let Some(ref mut gradients) = gradients {
                gradients.fill(0.0);

                if self.parameters.max_angular >= 1 {
                    let l = 1;
                    for n in 0..self.parameters.max_radial {
                        let gto_constant = self.gto_gaussian_constants[n];
                        let a = 0.5 * (n + l) as f64 + 1.5;
                        let b = 2.5;
                        let c_dn = (c + gto_constant).powf(-a);
                        let factor = global_factor * c * c_dn;

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
    use approx::{assert_relative_eq, assert_ulps_eq};

    use super::*;

    use super::super::{GtoRadialIntegral, GtoParameters, RadialIntegral};
    use ndarray::Array2;

    #[test]
    #[should_panic = "max_radial must be at least 1"]
    fn invalid_max_radial() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 0,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "cutoff must be a positive number"]
    fn negative_cutoff() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: -3.0,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "cutoff must be a positive number"]
    fn infinite_cutoff() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: std::f64::INFINITY,
            atomic_gaussian_width: 0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "atomic_gaussian_width must be a positive number"]
    fn negative_atomic_gaussian_width() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: -0.5
        }).unwrap();
    }

    #[test]
    #[should_panic = "atomic_gaussian_width must be a positive number"]
    fn infinite_atomic_gaussian_width() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 10,
            max_angular: 4,
            cutoff: 3.0,
            atomic_gaussian_width: std::f64::INFINITY,
        }).unwrap();
    }

    #[test]
    #[should_panic = "radial overlap matrix is singular, try with a lower max_radial (current value is 30)"]
    fn ill_conditioned_orthonormalization() {
        GtoRadialIntegral::new(GtoParameters {
            max_radial: 30,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
    }

    #[test]
    #[should_panic = "wrong size for values array, expected [4, 2] but got [3, 2]"]
    fn values_array_size() {
        let gto = GtoRadialIntegral::new(GtoParameters {
            max_radial: 2,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
        let mut values = Array2::from_elem((3, 2), 0.0);

        gto.compute(1.0, values.view_mut(), None);
    }

    #[test]
    #[should_panic = "wrong size for gradients array, expected [4, 2] but got [3, 2]"]
    fn gradient_array_size() {
        let gto = GtoRadialIntegral::new(GtoParameters {
            max_radial: 2,
            max_angular: 3,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();
        let mut values = Array2::from_elem((4, 2), 0.0);
        let mut gradients = Array2::from_elem((3, 2), 0.0);

        gto.compute(1.0, values.view_mut(), Some(gradients.view_mut()));
    }

    #[test]
    fn gradients_near_zero() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = GtoRadialIntegral::new(GtoParameters {
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
            gradients, gradients_plus, epsilon=1e-12, max_relative=1e-6,
        );
    }

    #[test]
    fn gto_finite_differences() {
        let max_radial = 8;
        let max_angular = 8;
        let gto = GtoRadialIntegral::new(GtoParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        let rij = 3.4;
        let delta = 1e-9;

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut values_delta = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        gto.compute(rij, values.view_mut(), Some(gradients.view_mut()));
        gto.compute(rij + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;

        assert_relative_eq!(
            finite_differences, gradients, max_relative=1e-4
        );
    }

    #[test]
    fn overlap() {
        // some basic sanity checks on the overlap matrix
        let max_radial = 8;
        let cutoff = 6.3;

        let gto_gaussian_widths = (0..max_radial).into_iter().map(|n| {
            let n = n as f64;
            let n_max = max_radial as f64;
            cutoff * f64::max(f64::sqrt(n), 1.0) / n_max
        }).collect::<Vec<_>>();

        let overlap = gto_overlap_matrix(
            max_radial,
            &gto_gaussian_widths,
        );

        for i in 0..max_radial {
            assert_ulps_eq!(overlap[(i, i)], 1.0);
        }

        for i in 0..max_radial {
            for j in i..max_radial {
                assert!(overlap[(j, i)] > 0.0);
            }
        }
    }
}
