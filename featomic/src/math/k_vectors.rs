//! Generate the k-vectors (also called reciprocal or Fourier vectors) needed
//! for the k-space implementation of LODE and SOAP. More specifically, these
//! are all points of a (reciprocal space) lattice that lie within a ball of a
//! specified cutoff radius.

use crate::Vector3D;
use crate::systems::{UnitCell, CellShape};

/// A single k-vector and its norm stored together
#[derive(Debug, Clone)]
pub struct KVector {
    /// direction of the k-vector (i.e. normalized vector)
    pub direction: Vector3D,
    /// length of the k-vector
    pub norm: f64,
}

/// Generate k-vectors up to a certain cutoff (in reciprocal space units)
/// for a given cell.
pub fn compute_k_vectors(cell: &UnitCell, k_cutoff: f64) -> Vec<KVector> {
    assert_ne!(cell.shape(), CellShape::Infinite, "can not compute k-vectors for an infinite cell");

    let cell = cell.matrix();
    let reciprocal_cell = 2.0 * std::f64::consts::PI * cell.transposed().inverse();

    let cutoff_squared = k_cutoff * k_cutoff;
    let b1 = Vector3D::from(reciprocal_cell[0]);
    let b2 = Vector3D::from(reciprocal_cell[1]);
    let b3 = Vector3D::from(reciprocal_cell[2]);

    let m = reciprocal_cell * reciprocal_cell.transposed(); // dot product
    let k_volume = reciprocal_cell.determinant();

    let n1_max = ((m[1][1] * m[2][2] - m[1][2] * m[1][2]).sqrt() / k_volume * k_cutoff) as isize;
    let n2_max = ((m[2][2] * m[0][0] - m[2][0] * m[2][0]).sqrt() / k_volume * k_cutoff) as isize;
    let n3_max = ((m[0][0] * m[1][1] - m[0][1] * m[0][1]).sqrt() / k_volume * k_cutoff) as isize;

    let mut results = Vec::new();

    for n3 in 1..n3_max + 1 {
        let k = n3 as f64 * b3;
        let norm_squared = k.norm2();
        if norm_squared < cutoff_squared {
            let norm = norm_squared.sqrt();
            results.push(KVector {
                direction: k / norm,
                norm: norm,
            });
        }
    }

    for n2 in 1..n2_max + 1 {
        for n3 in -n3_max..n3_max + 1 {
            let k = n2 as f64 * b2 + n3 as f64 * b3;
            let norm_squared = k.norm2();
            if norm_squared < cutoff_squared {
                let norm = norm_squared.sqrt();
                results.push(KVector {
                    direction: k / norm,
                    norm: norm,
                });
            }
        }
    }

    for n1 in 1..n1_max + 1 {
        for n2 in -n2_max..n2_max + 1 {
            for n3 in -n3_max..n3_max + 1 {
                let k = n1 as f64 * b1 + n2 as f64 * b2 + n3 as f64 * b3;
                let norm_squared = k.norm2();
                if norm_squared < cutoff_squared {
                    let norm = norm_squared.sqrt();
                    results.push(KVector {
                        direction: k / norm,
                        norm: norm,
                    });
                }
            }
        }
    }

    return results;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix3;

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
            cells.push(UnitCell::from(cell.inverse().transposed()));
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
            for (ik, &k_cutoff) in cutoffs.iter().enumerate() {
                let k_vectors = compute_k_vectors(&cell, k_cutoff);

                // Check whether number of obtained vectors agrees with exact result
                assert_eq!(k_vectors.len(), num_vectors_correct[ik]);

                // Check that the norms lie inside the cutoff ball
                for k_vector in k_vectors {
                    assert!(k_vector.norm < k_cutoff);
                }
            }
        }
    }
}
