//! The `UnitCell` type represents the enclosing box of a simulated system, with
//! some type of periodic condition.
use std::f64;
use crate::{Matrix3, Vector3D};

/// The shape of a cell determine how we will be able to compute the periodic
/// boundaries condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellShape {
    /// Infinite unit cell, with no boundaries
    Infinite,
    /// Orthorhombic unit cell, with cuboid shape
    Orthorhombic,
    /// Triclinic unit cell, with arbitrary parallelepiped shape
    Triclinic,
}

/// An `UnitCell` defines the system physical boundaries.
///
/// The shape of the cell can be any of the [`CellShape`][CellShape], and will
/// influence how periodic boundary conditions are applied.
///
/// [CellShape]: enum.CellShape.html
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitCell {
    /// Unit cell matrix
    matrix: Matrix3,
    /// Transpose of the unit cell matrix, cached from matrix
    transpose: Matrix3,
    /// Inverse of the transpose of the unit cell matrix, cached from matrix
    inverse: Matrix3,
    /// Unit cell shape
    shape: CellShape,
}

impl From<Matrix3> for UnitCell {
    fn from(matrix: Matrix3) -> UnitCell {
        if matrix == Matrix3::zero() {
            return UnitCell::infinite()
        }

        assert!(matrix.determinant() > 1e-6, "matrix is not invertible");

        let is_close_0 = |value| f64::abs(value) < 1e-6;
        let is_diagonal = |matrix: Matrix3| {
            is_close_0(matrix[0][1]) && is_close_0(matrix[0][2]) &&
            is_close_0(matrix[1][0]) && is_close_0(matrix[1][2]) &&
            is_close_0(matrix[2][0]) && is_close_0(matrix[2][1])
        };

        let shape = if is_diagonal(matrix) {
            CellShape::Orthorhombic
        } else {
            CellShape::Triclinic
        };

        return UnitCell {
            matrix: matrix,
            transpose: matrix.transposed(),
            inverse: matrix.transposed().inverse(),
            shape: shape
        }
    }
}

impl UnitCell {
    /// Create an infinite unit cell
    pub fn infinite() -> UnitCell {
        UnitCell {
            matrix: Matrix3::zero(),
            transpose: Matrix3::zero(),
            inverse: Matrix3::zero(),
            shape: CellShape::Infinite,
        }
    }

    /// Create an orthorhombic unit cell, with side lengths `a, b, c`.
    pub fn orthorhombic(a: f64, b: f64, c: f64) -> UnitCell {
        assert!(a > 0.0 && b > 0.0 && c > 0.0, "Cell lengths must be positive");
        let matrix = Matrix3::new([
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, c]
        ]);
        UnitCell {
            matrix: matrix,
            transpose: matrix,
            inverse: matrix.inverse(),
            shape: CellShape::Orthorhombic,
        }
    }

    /// Create a cubic unit cell, with side lengths `length, length, length`.
    pub fn cubic(length: f64) -> UnitCell {
        UnitCell::orthorhombic(length, length, length)
    }

    /// Create a triclinic unit cell, with side lengths `a, b, c` and angles
    /// `alpha, beta, gamma`.
    pub fn triclinic(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) -> UnitCell {
        assert!(a > 0.0 && b > 0.0 && c > 0.0, "Cell lengths must be positive");
        let cos_alpha = alpha.to_radians().cos();
        let cos_beta = beta.to_radians().cos();
        let (sin_gamma, cos_gamma) = gamma.to_radians().sin_cos();

        let b_x = b * cos_gamma;
        let b_y = b * sin_gamma;

        let c_x = c * cos_beta;
        let c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        let c_z = f64::sqrt(c * c - c_y * c_y - c_x * c_x);

        return UnitCell::from(Matrix3::new([
            [a,   0.0, 0.0],
            [b_x, b_y, 0.0],
            [c_x, c_y, c_z],
        ]));
    }

    /// Get the cell shape
    pub fn shape(&self) -> CellShape {
        self.shape
    }

    /// Check if this unit cell is infinite, *i.e.* if it does not have
    /// periodic boundary conditions.
    pub fn is_infinite(&self) -> bool {
        self.shape() == CellShape::Infinite
    }

    /// Get the first length of the cell (i.e. the norm of the first vector of
    /// the cell)
    pub fn a(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => Vector3D::from(self.matrix[0]).norm(),
            CellShape::Orthorhombic | CellShape::Infinite => self.matrix[0][0],
        }
    }

    /// Get the second length of the cell (i.e. the norm of the second vector of
    /// the cell)
    pub fn b(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => Vector3D::from(self.matrix[1]).norm(),
            CellShape::Orthorhombic | CellShape::Infinite => self.matrix[1][1],
        }
    }

    /// Get the third length of the cell (i.e. the norm of the third vector of
    /// the cell)
    pub fn c(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => Vector3D::from(self.matrix[2]).norm(),
            CellShape::Orthorhombic | CellShape::Infinite => self.matrix[2][2],
        }
    }

    /// Get the distances between faces of the unit cell
    pub fn distances_between_faces(&self) -> Vector3D {
        if self.shape == CellShape::Infinite {
            return Vector3D::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        }

        let a = Vector3D::from(self.matrix[0]);
        let b = Vector3D::from(self.matrix[1]);
        let c = Vector3D::from(self.matrix[2]);

        // Plans normal vectors
        let na = (b ^ c).normalized();
        let nb = (c ^ a).normalized();
        let nc = (a ^ b).normalized();

        Vector3D::new(f64::abs(na * a), f64::abs(nb * b), f64::abs(nc * c))
    }

    /// Get the first angle of the cell
    pub fn alpha(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => {
                let b = Vector3D::from(self.matrix[1]);
                let c = Vector3D::from(self.matrix[2]);
                angle(b, c).to_degrees()
            }
            CellShape::Orthorhombic | CellShape::Infinite => 90.0,
        }
    }

    /// Get the second angle of the cell
    pub fn beta(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => {
                let a = Vector3D::from(self.matrix[0]);
                let c = Vector3D::from(self.matrix[2]);
                angle(a, c).to_degrees()
            }
            CellShape::Orthorhombic | CellShape::Infinite => 90.0,
        }
    }

    /// Get the third angle of the cell
    pub fn gamma(&self) -> f64 {
        match self.shape {
            CellShape::Triclinic => {
                let a = Vector3D::from(self.matrix[0]);
                let b = Vector3D::from(self.matrix[1]);
                angle(a, b).to_degrees()
            }
            CellShape::Orthorhombic | CellShape::Infinite => 90.0,
        }
    }

    /// Get the volume of the cell
    pub fn volume(&self) -> f64 {
        let volume = self.matrix.determinant();
        assert!(volume >= 0.0, "Volume is not positive!");
        return volume;
    }

    /// Get the matricial representation of the unit cell
    pub fn matrix(&self) -> Matrix3 {
        self.matrix
    }

    /// Get the fractional representation of the `vector` in this cell
    pub fn fractional(&self, vector: Vector3D) -> Vector3D {
        // this needs to use the inverse of the transpose of the matrix, since
        // we only have code to multiply a vector by a matrix on the left
        return self.inverse * vector;
    }

    /// Get the Cartesian representation of the `fractional` vector in this
    /// cell
    pub fn cartesian(&self, fractional: Vector3D) -> Vector3D {
        // this needs to use the inverse of the transpose of the matrix, since
        // we only have code to multiply a vector by a matrix on the left
        return self.transpose * fractional;
    }
}

/// Get the angles between the vectors `u` and `v`.
fn angle(u: Vector3D, v: Vector3D) -> f64 {
    let un = u.normalized();
    let vn = v.normalized();
    f64::acos(un * vn)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::{assert_ulps_eq, assert_relative_eq};

    #[test]
    #[should_panic(expected = "Cell lengths must be positive")]
    fn negative_cubic() {
        let _ = UnitCell::cubic(-4.0);
    }

    #[test]
    #[should_panic(expected = "Cell lengths must be positive")]
    fn negative_ortho() {
        let _ = UnitCell::orthorhombic(3.0, 0.0, -5.0);
    }

    #[test]
    #[should_panic(expected = "Cell lengths must be positive")]
    fn negative_triclinic() {
        let _ = UnitCell::triclinic(3.0, 0.0, -5.0, 90.0, 90.0, 90.0);
    }

    #[test]
    fn infinite() {
        let cell = UnitCell::infinite();
        assert_eq!(cell.shape(), CellShape::Infinite);
        assert!(cell.is_infinite());

        assert_eq!(cell.a(), 0.0);
        assert_eq!(cell.b(), 0.0);
        assert_eq!(cell.c(), 0.0);

        assert_eq!(cell.alpha(), 90.0);
        assert_eq!(cell.beta(), 90.0);
        assert_eq!(cell.gamma(), 90.0);

        assert_eq!(cell.volume(), 0.0);
    }

    #[test]
    fn cubic() {
        let cell = UnitCell::cubic(3.0);
        assert_eq!(cell.shape(), CellShape::Orthorhombic);
        assert!(!cell.is_infinite());

        assert_eq!(cell.a(), 3.0);
        assert_eq!(cell.b(), 3.0);
        assert_eq!(cell.c(), 3.0);

        assert_eq!(cell.alpha(), 90.0);
        assert_eq!(cell.beta(), 90.0);
        assert_eq!(cell.gamma(), 90.0);

        assert_eq!(cell.volume(), 3.0 * 3.0 * 3.0);
    }

    #[test]
    fn orthorhombic() {
        let cell = UnitCell::orthorhombic(3.0, 4.0, 5.0);
        assert_eq!(cell.shape(), CellShape::Orthorhombic);
        assert!(!cell.is_infinite());

        assert_eq!(cell.a(), 3.0);
        assert_eq!(cell.b(), 4.0);
        assert_eq!(cell.c(), 5.0);

        assert_eq!(cell.alpha(), 90.0);
        assert_eq!(cell.beta(), 90.0);
        assert_eq!(cell.gamma(), 90.0);

        assert_eq!(cell.volume(), 3.0 * 4.0 * 5.0);
    }

    #[test]
    fn triclinic() {
        let cell = UnitCell::triclinic(3.0, 4.0, 5.0, 80.0, 90.0, 110.0);
        assert_eq!(cell.shape(), CellShape::Triclinic);
        assert!(!cell.is_infinite());

        assert_eq!(cell.a(), 3.0);
        assert_eq!(cell.b(), 4.0);
        assert_eq!(cell.c(), 5.0);

        assert_eq!(cell.alpha(), 80.0);
        assert_eq!(cell.beta(), 90.0);
        assert_eq!(cell.gamma(), 110.0);

        assert_relative_eq!(cell.volume(), 55.410529, epsilon = 1e-6);
    }

    #[test]
    fn distances_between_faces() {
        let ortho = UnitCell::orthorhombic(3.0, 4.0, 5.0);
        assert_eq!(ortho.distances_between_faces(), Vector3D::new(3.0, 4.0, 5.0));

        let triclinic = UnitCell::triclinic(3.0, 4.0, 5.0, 90.0, 90.0, 90.0);
        assert_eq!(triclinic.distances_between_faces(), Vector3D::new(3.0, 4.0, 5.0));

        let triclinic = UnitCell::triclinic(3.0, 4.0, 5.0, 90.0, 80.0, 100.0);
        assert_eq!(triclinic.distances_between_faces(), Vector3D::new(2.908132319388713, 3.9373265973230853, 4.921658246653857));
    }

    #[test]
    fn fractional_cartesian() {
        let cell = UnitCell::cubic(5.0);

        assert_eq!(
            cell.fractional(Vector3D::new(0.0, 10.0, 4.0)),
            Vector3D::new(0.0, 2.0, 0.8)
        );
        assert_eq!(
            cell.cartesian(Vector3D::new(0.0, 2.0, 0.8)),
            Vector3D::new(0.0, 10.0, 4.0)
        );

        let cell = UnitCell::triclinic(5.0, 6.0, 3.6, 90.0, 53.0, 77.0);
        let tests = vec![
            Vector3D::new(0.0, 10.0, 4.0),
            Vector3D::new(-5.0, 12.0, 4.9),
        ];

        for test in tests {
            let transformed = cell.cartesian(cell.fractional(test));
            assert_ulps_eq!(test, transformed, epsilon = 1e-15);
        }
    }
}
