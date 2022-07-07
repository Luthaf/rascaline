// Generate the k-vectors (also called reciprocal or Fourier vectors)
// needed for the k-space implementation of LODE and SOAP.
// More specifically, these are all points of a (reciprocal space)
// lattice that lie within a ball of a specified cutoff radius.

use crate::{Error, Vector3D, Matrix3};

/// Kvectors
#[derive(Debug, Clone)]
pub struct KVector {
    /// 3 component k-vector
    pub vector: Vector3D,
    /// length of the k-vector
    pub norm: f64
}

pub fn compute_kvectors(cell: &Matrix3, kcutoff: f64) -> Vec<KVector> {
    // Generate k-vectors up to a certain cutoff for a given cell
    let reciprocal_cell = 2.0 * std::f64::consts::PI * cell.inverse();

    let cutoff_squared = kcutoff * kcutoff;
    let b1 = Vector3D::from(reciprocal_cell[0]);
    let b2 = Vector3D::from(reciprocal_cell[1]);
    let b3 = Vector3D::from(reciprocal_cell[2]);

    let m = reciprocal_cell * reciprocal_cell.transposed(); // dot product
    let kvol = reciprocal_cell.determinant();

    let n1max = ((m[1][1] * m[2][2] - m[1][2] * m[1][2]) / kvol * kcutoff).sqrt().floor() as isize;
    let n2max = ((m[2][2] * m[0][0] - m[2][0] * m[2][0]) / kvol * kcutoff).sqrt().floor() as isize;
    let n3max = ((m[0][0] * m[1][1] - m[0][1] * m[0][1]) / kvol * kcutoff).sqrt().floor() as isize;

    let mut kvecs = Vec::new();

    for n3 in 1..n3max + 1 {
        let k = n3 as f64 * b3;
        let norm_squared = k.norm2();
        if norm_squared < cutoff_squared {
            kvecs.push(KVector{vector: k, norm: norm_squared.sqrt()});
        }
    }

    for n2 in 1..n2max + 1 {
        for n3 in -n3max..n3max + 1 {
            let k = n2 as f64 * b2 + n3 as f64 * b3;
            let norm_squared = k.norm2();
            if norm_squared < cutoff_squared {
                kvecs.push(KVector{vector: k, norm: norm_squared.sqrt()});
            }
        }
    }

    for n1 in 1..n1max+1 {
        for n2 in -n2max..n2max+1 {
            for n3 in -n3max..n3max+1 {
                let k = n1 as f64 * b1 + n2 as f64 * b2 + n3 as f64 * b3;
                let norm_squared = k.norm2();
                if norm_squared < cutoff_squared {
                    kvecs.push(KVector{vector: k, norm: norm_squared.sqrt()});
                }
            }
        }
    }

    kvecs
}