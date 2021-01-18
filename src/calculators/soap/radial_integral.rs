use ndarray::{Array2, ArrayViewMut2};

use crate::math::gamma;
use super::{HyperGeometricSphericalExpansion, HyperGeometricParameters};

const PI_TO_THREE_HALF: f64 = 15.503138340149908;

pub trait RadialIntegral: std::panic::RefUnwindSafe {
    /// Compute the radial integral for a single atomic distance `rij` and store
    /// the resulting data in the `max_radial x max_angular` array `values`. If
    /// `gradients` is `Some`, also compute and store gradients.
    fn compute(&self, rij: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);
}

/// Parameters controlling GTO radial basis
#[derive(Debug, Clone, Copy)]
pub struct GTOParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// atomic density gaussian width
    pub atomic_gaussian_width: f64,
    /// cutoff radius
    pub cutoff: f64,
}

impl GTOParameters {
    fn validate(&self) {
        assert!(self.max_radial > 0, "max_radial must be at least 1");

        assert!(
            self.cutoff > 0.0 && self.cutoff.is_finite(),
            "cutoff must be a positive number"
        );

        assert!(
            self.atomic_gaussian_width > 0.0 && self.atomic_gaussian_width.is_finite(),
            "atomic_gaussian_width must be a positive number"
        );
    }
}

#[derive(Debug, Clone)]
pub struct GTO {
    parameters: GTOParameters,
    hypergeometric: HyperGeometricSphericalExpansion,
    /// 1/2σ^2, with σ the atomic density gaussian width
    atomic_gaussian_constant: f64,
    /// 1/2σ_n^2, with σ_n the GTO gaussian width, i.e. `cutoff * max(√n, 1) / (n_max + 1) `
    gto_gaussian_constants: Vec<f64>,
    /// `sqrt[ 2 / (σ_n^{2n+3} * Γ(n + 3/2) ) ]`
    gaussian_normalization: Vec<f64>,
}

impl GTO {
    pub fn new(parameters: GTOParameters) -> GTO {
        parameters.validate();

        let gto_gaussian_widths = (0..parameters.max_radial).into_iter().map(|n| {
            let n = n as f64;
            let n_max = parameters.max_radial as f64;
            parameters.cutoff * f64::max(f64::sqrt(n), 1.0) / (n_max + 1.0)
        }).collect::<Vec<_>>();

        let gto_gaussian_constants = gto_gaussian_widths.iter()
            .map(|&sigma| 1.0 / (2.0 * sigma * sigma))
            .collect::<Vec<_>>();

        let gaussian_normalization = gto_gaussian_widths.iter()
            .zip(0..parameters.max_radial)
            .map(|(sigma, n)| f64::sqrt(2.0 / (sigma.powi(2 * n as i32 + 3) * gamma(n as f64 + 1.5))))
            .collect::<Vec<_>>();

        let hypergeometric = HyperGeometricSphericalExpansion::new(parameters.max_radial, parameters.max_angular);

        let sigma2 = parameters.atomic_gaussian_width * parameters.atomic_gaussian_width;
        return GTO {
            parameters: parameters,
            hypergeometric: hypergeometric,
            atomic_gaussian_constant: 1.0 / (2.0 * sigma2),
            gto_gaussian_constants: gto_gaussian_constants,
            gaussian_normalization: gaussian_normalization,
        }
    }
}

impl RadialIntegral for GTO {
    fn compute(&self, rij: f64, mut values: ArrayViewMut2<f64>, mut gradients: Option<ArrayViewMut2<f64>>) {
        assert_eq!(values.shape(), [self.parameters.max_radial, self.parameters.max_angular + 1]);

        if let Some(ref gradients) = gradients {
            assert_eq!(gradients.shape(), [self.parameters.max_radial, self.parameters.max_angular + 1]);
        }

        let hyperg_parameters = HyperGeometricParameters {
            atomic_gaussian_constant: self.atomic_gaussian_constant,
            gto_gaussian_constants: &self.gto_gaussian_constants,
        };
        self.hypergeometric.compute(rij, hyperg_parameters, values.view_mut(), gradients.as_mut().map(|g| g.view_mut()));

        let mut factors = Array2::from_elem((self.parameters.max_radial, self.parameters.max_angular + 1), 0.0);
        let mut grad_factors = Array2::from_elem((self.parameters.max_radial, self.parameters.max_angular + 1), 0.0);
        let c = self.atomic_gaussian_constant;
        for n in 0..self.parameters.max_radial {
            let gaussian_normalization = self.gaussian_normalization[n];
            let gto_constant = self.gto_gaussian_constants[n];
            for l in 0..(self.parameters.max_angular + 1) {
                let n_l_3_over_2 = 0.5 * (n + l) as f64 + 1.5;
                let c_l_rij_l = c.powi(l as i32) * rij.powi(l as i32);
                let c_dn = (c + gto_constant).powf(-n_l_3_over_2);

                factors[[n, l]] = PI_TO_THREE_HALF * gaussian_normalization * c_l_rij_l * c_dn;
                grad_factors[[n, l]] = l as f64 / rij;
            }
        }

        values *= &factors;

        if let Some(ref mut gradients) = gradients {
            *gradients *= &factors;
            grad_factors *= &values;
            *gradients += &grad_factors;
        }
    }
}

#[cfg(test)]
mod tests {
    mod gto {
        use approx::assert_relative_eq;

        use super::super::*;

        #[test]
        #[should_panic = "max_radial must be at least 1"]
        fn invalid_max_radial() {
            GTO::new(GTOParameters {
                max_radial: 0,
                max_angular: 4,
                cutoff: 3.0,
                atomic_gaussian_width: 0.5
            });
        }

        #[test]
        #[should_panic = "cutoff must be a positive number"]
        fn negative_cutoff() {
            GTO::new(GTOParameters {
                max_radial: 10,
                max_angular: 4,
                cutoff: -3.0,
                atomic_gaussian_width: 0.5
            });
        }

        #[test]
        #[should_panic = "cutoff must be a positive number"]
        fn infiniye_cutoff() {
            GTO::new(GTOParameters {
                max_radial: 10,
                max_angular: 4,
                cutoff: f64::INFINITY,
                atomic_gaussian_width: 0.5
            });
        }

        #[test]
        #[should_panic = "atomic_gaussian_width must be a positive number"]
        fn negative_atomic_gaussian_width() {
            GTO::new(GTOParameters {
                max_radial: 10,
                max_angular: 4,
                cutoff: 3.0,
                atomic_gaussian_width: -0.5
            });
        }

        #[test]
        #[should_panic = "atomic_gaussian_width must be a positive number"]
        fn infinite_atomic_gaussian_width() {
            GTO::new(GTOParameters {
                max_radial: 10,
                max_angular: 4,
                cutoff: 3.0,
                atomic_gaussian_width: f64::INFINITY,
            });
        }

        #[test]
        fn gto_finite_differences() {
            let max_radial = 10;
            let max_angular = 4;
            let gto = GTO::new(GTOParameters {
                max_radial: 10,
                max_angular: 4,
                cutoff: 5.0,
                atomic_gaussian_width: 0.5,
            });

            let rij = 3.4;
            let delta = 1e-9;

            let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);
            let mut values_delta = Array2::from_elem((max_radial, max_angular + 1), 0.0);
            let mut gradients = Array2::from_elem((max_radial, max_angular + 1), 0.0);
            gto.compute(rij, values.view_mut(), Some(gradients.view_mut()));
            gto.compute(rij + delta, values_delta.view_mut(), None);

            let finite_differences = (&values_delta - &values) / delta;

            for n in 0..max_radial {
                for l in 0..(max_angular + 1) {
                    assert_relative_eq!(
                        finite_differences[[n, l]], gradients[[n, l]],
                        epsilon=1e-5, max_relative=1e-5
                    );
                }
            }
        }
    }
}
