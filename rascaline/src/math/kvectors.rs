// Generate the k-vectors (also called reciprocal or Fourier vectors)
// needed for the k-space implementation of LODE and SOAP.
// More specifically, these are all points of a (reciprocal space)
// lattice that lie within a ball of a specified cutoff radius.
use crate::{Matrix3, Vector3D};

/// Kvectors
#[derive(Debug, Clone)]
pub struct KVector {
    /// 3 component k-vector
    pub vector: Vector3D,
    /// length of the k-vector
    pub norm: f64,
}

/// Generate k-vectors up to a certain cutoff (in reciprocal space units) 
/// for a given cell.
pub fn compute_kvectors(cell: &Matrix3, kcutoff: f64) -> Vec<KVector> {
    debug_assert_ne!(cell, &Matrix3::zero(), "Invalid cell matrix!");

    let reciprocal_cell = 2.0 * std::f64::consts::PI * cell.transposed().inverse();

    let cutoff_squared = kcutoff * kcutoff;
    let b1 = Vector3D::from(reciprocal_cell[0]);
    let b2 = Vector3D::from(reciprocal_cell[1]);
    let b3 = Vector3D::from(reciprocal_cell[2]);

    let m = reciprocal_cell * reciprocal_cell.transposed(); // dot product
    let kvol = reciprocal_cell.determinant();

    let n1max = ((m[1][1] * m[2][2] - m[1][2] * m[1][2]).sqrt() / kvol * kcutoff) as isize;
    let n2max = ((m[2][2] * m[0][0] - m[2][0] * m[2][0]).sqrt() / kvol * kcutoff) as isize;
    let n3max = ((m[0][0] * m[1][1] - m[0][1] * m[0][1]).sqrt() / kvol * kcutoff) as isize;

    let mut kvecs = Vec::new();

    for n3 in 1..n3max + 1 {
        let k = n3 as f64 * b3;
        let norm_squared = k.norm2();
        if norm_squared < cutoff_squared {
            kvecs.push(KVector {
                vector: k,
                norm: norm_squared.sqrt(),
            });
        }
    }

    for n2 in 1..n2max + 1 {
        for n3 in -n3max..n3max + 1 {
            let k = n2 as f64 * b2 + n3 as f64 * b3;
            let norm_squared = k.norm2();
            if norm_squared < cutoff_squared {
                kvecs.push(KVector {
                    vector: k,
                    norm: norm_squared.sqrt(),
                });
            }
        }
    }

    for n1 in 1..n1max + 1 {
        for n2 in -n2max..n2max + 1 {
            for n3 in -n3max..n3max + 1 {
                let k = n1 as f64 * b1 + n2 as f64 * b2 + n3 as f64 * b3;
                let norm_squared = k.norm2();
                if norm_squared < cutoff_squared {
                    kvecs.push(KVector {
                        vector: k,
                        norm: norm_squared.sqrt(),
                    });
                }
            }
        }
    }

    kvecs
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SQRT_3: f64 = 1.7320508075688772;
    const SQRT_5: f64 = 2.23606797749979;

    #[test]
    fn test_kvectors() {
        let cell_cubic = Matrix3::one();
        let cell_triclinic = Matrix3::new([[1.0, 0.0, 0.0], [5.0, 1.0, 0.0], [3.0, 4.0, 1.0]]);
        // Generated with NumPy
        // Q = np.linalg.qr(np.random.normal(0, 1, (3, 3)))[0]
        // cell_rotated = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]]) @ Q
        let cell_rotated = Matrix3::new([
            [-0.10740572, -0.73747025, -0.66678455],
            [0.30874359, -3.40258247, -3.7851169],
            [3.58349326, -1.68574423, -3.21198419],
        ]);

        let reciprocal_cells = [cell_cubic, cell_triclinic, cell_rotated];
        let mut cells = Vec::new();
        for reciprocal_cells in reciprocal_cells {
            let cell = reciprocal_cells / (2.0 * std::f64::consts::PI);
            cells.push(cell.inverse().transposed());
        }
        let num_vectors_correct = [3, 3, 9, 9, 13, 13, 16];

        let eps = 1e-2;
        let cutoffs = [
            1.0 + eps,
            std::f64::consts::SQRT_2 - eps,
            std::f64::consts::SQRT_2 + eps,
            SQRT_3 - eps,
            SQRT_3 + eps,
            2.0 - eps,
            SQRT_5 - eps,
        ];

        for cell in cells {
            for (ik, kcut) in cutoffs.iter().enumerate() {
                let kvecs = compute_kvectors(&cell, *kcut);

                // Check whether number of obtained vectors agrees with exact result
                assert!(kvecs.len() == num_vectors_correct[ik]);

                // Check that the obtained normes are indeed the norms of the
                // corresponding k-vectors and that they lie in the cutoff ball
                for kvec in kvecs {
                    assert_relative_eq!(kvec.norm, kvec.vector.norm(), max_relative = 1e-7);
                    assert!(kvec.norm < *kcut);
                }
            }
        }
    }
}
