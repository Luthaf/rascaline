use ndarray::{Array1, Array2};

use crate::math::gamma;

#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// Use a radial basis similar to Gaussian-Type Orbitals.
///
/// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
/// = cutoff * \sqrt{n} / n_max`
pub struct GtoRadialBasis {
    pub max_radial: usize,
    pub cutoff: f64,
}

impl GtoRadialBasis {
    /// Get the overlap matrix between non-orthonormalized GTO basis function
    pub fn overlap(&self) -> Array2<f64> {
        let gaussian_widths = self.gaussian_widths();
        let n_max = self.max_radial;

        let mut overlap = Array2::from_elem((n_max, n_max), 0.0);
        for n1 in 0..n_max {
            let sigma1 = gaussian_widths[n1];
            let sigma1_sq = sigma1 * sigma1;
            for n2 in n1..n_max {
                let sigma2 = gaussian_widths[n2];
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

    /// Get the vector of GTO Gaussian width, i.e. `cutoff * max(√n, 1) / n_max`
    pub fn gaussian_widths(&self) -> Vec<f64> {
        return (0..self.max_radial).map(|n| {
            let n = n as f64;
            let n_max = self.max_radial as f64;
            self.cutoff * f64::max(f64::sqrt(n), 1.0) / n_max
        }).collect();
    }

    /// Get the matrix to orthonormalize the GTO basis
    pub fn orthonormalization_matrix(&self) -> Array2<f64> {
        let normalization = self.gaussian_widths().iter()
            .zip(0..self.max_radial)
            .map(|(sigma, n)| f64::sqrt(2.0 / (sigma.powi(2 * n as i32 + 3) * gamma(n as f64 + 1.5))))
            .collect::<Array1<_>>();

        let overlap = self.overlap();
        // compute overlap^-1/2 through its eigendecomposition
        let mut eigen = crate::math::SymmetricEigen::new(overlap);
        for n in 0..self.max_radial {
            if eigen.eigenvalues[n] <= f64::EPSILON {
                panic!(
                    "radial overlap matrix is singular, try with a lower \
                    max_radial (current value is {})", self.max_radial
                );
            }
            eigen.eigenvalues[n] = 1.0 / f64::sqrt(eigen.eigenvalues[n]);
        }

        let orthonormalization = eigen.recompose().dot(&Array2::from_diag(&normalization));

        return orthonormalization;
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use super::*;

    #[test]
    fn gto_overlap() {
        // some basic sanity checks on the overlap matrix
        let basis = GtoRadialBasis {
            max_radial: 8,
            cutoff: 6.3,
        };

        let overlap = basis.overlap();

        for i in 0..basis.max_radial {
            for j in 0..basis.max_radial {
                if i == j {
                    assert_ulps_eq!(overlap[(i, j)], 1.0, max_ulps=10);
                } else {
                    assert!(overlap[(i, j)] > 0.0);
                    assert!(overlap[(i, j)] < 1.0);
                }
            }
        }
    }
}
