use ndarray::{Array2, ArrayViewMut2};

use nalgebra as na;
use nalgebra::linalg::SymmetricEigen;

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

fn permute_columns<T>(mat: &mut na::DMatrix<f64>, perm: &[(T, usize)]) {
    assert!(mat.ncols() == perm.len());
    let n = mat.ncols();
    let mut already_permuted = vec![0; n];
    for i in 0..n {
        if already_permuted[i] == 0 {
            let (mut j, mut k) = (i, perm[i].1);
            while k != i {
                mat.swap_columns(j, k);
                already_permuted[k] = 1;
                j = k;
                k = perm[k].1;
            }
            already_permuted[i] = 1;
        }
    }
}

// We have to sort eigenvalues because this is not done by default,
// see https://github.com/dimforge/nalgebra/issues/349
fn sort_eigen(eigen: &mut SymmetricEigen<f64, na::Dynamic>) {
    let mut s: Vec<(_, _)> = eigen
        .eigenvalues
        .into_iter()
        .enumerate()
        .map(|(idx, &v)| (v, idx))
        .collect();
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    permute_columns(&mut eigen.eigenvectors, &s);
    eigen.eigenvalues = na::DVector::from_vec(s.into_iter().map(|(v, _)| v).collect());
}

fn sorted_eigen(mat: na::DMatrix<f64>) -> SymmetricEigen<f64, na::Dynamic> {
    let mut eigen = mat.symmetric_eigen();
    sort_eigen(&mut eigen);
    return eigen;
}

#[derive(Debug, Clone)]
pub struct GTO {
    parameters: GTOParameters,
    hypergeometric: HyperGeometricSphericalExpansion,
    /// 1/2σ^2, with σ the atomic density gaussian width
    atomic_gaussian_constant: f64,
    /// 1/2σ_n^2, with σ_n the GTO gaussian width, i.e. `cutoff * max(√n, 1) / (n_max + 1) `
    gto_gaussian_constants: Vec<f64>,
    /// `n_max * n_max` matrix to orthonormalize the GTO basis
    gto_orthonormalization: Array2<f64>,
}

impl GTO {
    pub fn new(parameters: GTOParameters) -> GTO {
        parameters.validate();

        let gto_gaussian_widths = (0..parameters.max_radial).into_iter().map(|n| {
            let n = n as f64;
            let n_max = parameters.max_radial as f64;
            parameters.cutoff * f64::max(f64::sqrt(n), 1.0) / n_max
        }).collect::<Vec<_>>();

        let gto_gaussian_constants = gto_gaussian_widths.iter()
            .map(|&sigma| 1.0 / (2.0 * sigma * sigma))
            .collect::<Vec<_>>();

        let gaussian_normalization = gto_gaussian_widths.iter()
            .zip(0..parameters.max_radial)
            .map(|(sigma, n)| PI_TO_THREE_HALF * 0.25 * f64::sqrt(2.0 / (sigma.powi(2 * n as i32 + 3) * gamma(n as f64 + 1.5))))
            .collect::<Vec<_>>();

        let mut overlap = na::DMatrix::from_element(
            parameters.max_radial, parameters.max_radial, 0.0
        );

        for n1 in 0..parameters.max_radial {
            let sigma1 = gto_gaussian_widths[n1];
            let sigma1_sq = sigma1 * sigma1;
            for n2 in n1..parameters.max_radial {
                let sigma2 = gto_gaussian_widths[n2];
                let sigma2_sq = sigma2 * sigma2;

                let n1_n2_3_over_2 = 0.5 * (3.0 + n1 as f64 + n2 as f64);
                let value =
                    (0.5 / sigma1_sq + 0.5 / sigma2_sq).powf(-n1_n2_3_over_2)
                    / (sigma1.powi(n1 as i32) * sigma2.powi(n2 as i32))
                    * gamma(n1_n2_3_over_2)
                    / ((sigma1 * sigma2).powf(1.5) * f64::sqrt(gamma(n1 as f64 + 1.5) * gamma(n2 as f64 + 1.5)));

                overlap[(n1, n2)] = value;
                overlap[(n2, n1)] = value;
            }
        }

        // TODO: this is a over allocating mess
        // compute normalization * overlap^-1/2
        let mut eigen = sorted_eigen(overlap); // .symmetric_eigen();
        for n in 0..parameters.max_radial {
            if eigen.eigenvalues[n] <= 0.0 {
                panic!(
                    "radial overlap matrix is singular, try with a lower \
                    max_angular (current value is {})", parameters.max_angular
                );
            }
            eigen.eigenvalues[n] = 1.0 / f64::sqrt(eigen.eigenvalues[n]);
        }

        let na_gaussian_normalization = na::DMatrix::from_diagonal(
            &na::Vector::from(gaussian_normalization)
        );

        let gto_orthonormalization = na_gaussian_normalization * eigen.recompose();

        let gto_orthonormalization = Array2::from_shape_vec(
            (parameters.max_radial, parameters.max_radial),
            gto_orthonormalization.data.as_vec().clone()
        ).expect("wrong matrix size for gto_orthonormalization");
        // TODO end of mess

        let hypergeometric = HyperGeometricSphericalExpansion::new(parameters.max_radial, parameters.max_angular);

        let sigma2 = parameters.atomic_gaussian_width * parameters.atomic_gaussian_width;
        return GTO {
            parameters: parameters,
            hypergeometric: hypergeometric,
            atomic_gaussian_constant: 1.0 / (2.0 * sigma2),
            gto_gaussian_constants: gto_gaussian_constants,
            gto_orthonormalization: gto_orthonormalization,
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

        // TODO: check with a loop instead of allocating memory
        let mut factors = Array2::from_elem((self.parameters.max_radial, self.parameters.max_angular + 1), 0.0);
        let mut grad_factors = Array2::from_elem((self.parameters.max_radial, self.parameters.max_angular + 1), 0.0);
        let c = self.atomic_gaussian_constant;
        for n in 0..self.parameters.max_radial {
            let gto_constant = self.gto_gaussian_constants[n];
            for l in 0..(self.parameters.max_angular + 1) {
                let n_l_3_over_2 = 0.5 * (n + l) as f64 + 1.5;
                let c_l_rij_l = c.powi(l as i32) * rij.powi(l as i32);
                let c_dn = (c + gto_constant).powf(-n_l_3_over_2);

                factors[[n, l]] = c_l_rij_l * c_dn;
                grad_factors[[n, l]] = l as f64 / rij;
            }
        }

        values *= &factors;

        if let Some(ref mut gradients) = gradients {
            *gradients *= &factors;
            grad_factors *= &values;
            *gradients += &grad_factors;

            // do `gradients = self.gto_orthonormalization * gradients` without
            // introducing a temporary matrix
            gradients.t().dot(&self.gto_orthonormalization.t());
        }

        // do `values = self.gto_orthonormalization * values` without
        // introducing a temporary matrix
        values.t().dot(&self.gto_orthonormalization.t());
    }
}

#[cfg(test)]
mod tests {
    mod gto {
        use approx::assert_relative_eq;

        use super::super::{GTO, GTOParameters, RadialIntegral};
        use ndarray::Array2;

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
