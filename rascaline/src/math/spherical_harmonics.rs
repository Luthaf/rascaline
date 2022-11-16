#![allow(clippy::range_plus_one)]

use std::f64;
use std::f64::consts::SQRT_2;

use ndarray::ArrayView1;

use crate::Vector3D;

/// `\sqrt{\frac{1}{2 \pi}}`
const SQRT_1_OVER_2PI: f64 = 0.3989422804014327;
/// `\sqrt{3}`
const SQRT_3: f64 = 1.7320508075688772;
/// `\sqrt{3 / 2}`
const SQRT_3_OVER_2: f64 = 1.224744871391589;

/// Array storing data for `0 <= l <= l_max`, `0 <= m <= l`. This type
/// implements `Index<[usize; 2]>` and `IndexMut<[usize; 2]>` to allow writing
/// code like
///
/// ```ignore
/// let mut array = LegendreArray::new(8);
/// array[[6, 3]] = 3.0;
///
/// // this is an error m > l
/// array[[6, 7]]
/// // this is an error l > l_max
/// array[[9, 7]]
/// ```
#[derive(Clone)]
struct LegendreArray {
    max_angular: usize,
    data: Vec<f64>,
}

impl LegendreArray {
    /// Create a new `LegendreArray` with the given maximal angular degree, and
    /// all elements set to zero.
    pub fn new(max_angular: usize) -> LegendreArray {
        let size = (max_angular + 1) * (max_angular + 2) / 2;
        LegendreArray {
            max_angular: max_angular,
            data: vec![0.0; size],
        }
    }

    #[inline]
    fn linear_index(&self, index: [usize; 2]) -> usize {
        let [l, m] = index;
        debug_assert!(l <= self.max_angular && m <= l);
        return m + l * (l + 1) / 2;
    }
}

impl std::ops::Index<[usize; 2]> for LegendreArray {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &f64 {
        &self.data[self.linear_index(index)]
    }
}

impl std::ops::IndexMut<[usize; 2]> for LegendreArray {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut f64 {
        let i = self.linear_index(index);
        &mut self.data[i]
    }
}

impl std::fmt::Debug for LegendreArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LegendreArray[\n  l \\ m  ")?;
        for m in 0..(self.max_angular + 1) {
            write!(f, " {: ^12}", m)?;
        }
        writeln!(f)?;
        for l in 0..(self.max_angular + 1) {
            write!(f, "  {: <8}", l)?;
            for m in 0..=l {
                write!(f, " {:+.9}", self[[l, m]])?;
            }
            writeln!(f)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Array storing data for `0 <= l <= l_max`, `-l <= m <= l`. This type
/// implements `Index<[isize; 2]>` and `IndexMut<[isize; 2]>` to allow writing
/// code like
///
/// ```
/// # use rascaline::math::SphericalHarmonicsArray;
/// let mut array = SphericalHarmonicsArray::new(8);
/// array[[6, 3]] = 3.0;
/// array[[6, -3]] = -3.0;
///
/// // this is an error |m| > l
/// // array[[6, 7]] = 1.0;
/// // array[[6, -7]] = 1.0;
///
/// // this is an error l > l_max
/// // array[[9, 7]] = 1.0;
/// ```
#[derive(Clone)]
pub struct SphericalHarmonicsArray {
    max_angular: isize,
    data: Vec<f64>,
}

impl SphericalHarmonicsArray {
    /// Create a new `SphericalHarmonicsArray` with the given maximal angular
    /// degree, and all elements set to zero.
    pub fn new(max_angular: usize) -> SphericalHarmonicsArray {
        let size = (max_angular + 1) * (max_angular + 1);
        SphericalHarmonicsArray {
            max_angular: max_angular as isize,
            data: vec![0.0; size],
        }
    }

    #[inline]
    #[allow(clippy::suspicious_operation_groupings)]
    fn linear_index(&self, index: [isize; 2]) -> usize {
        let [l, m] = index;
        debug_assert!(l <= self.max_angular && -l <= m && m <= l);
        return (m + l + (l * l)) as usize;
    }

    /// Get the slice of the full array containing values for a given `l`. The
    /// size of the resulting view is `2 * l + 1`, and contains value for `m`
    /// from `-l` to `l` in order.
    #[inline]
    pub fn slice(&self, l: isize) -> ArrayView1<'_, f64> {
        let start = self.linear_index([l, -l]);
        let stop = self.linear_index([l, l]);
        return ArrayView1::from(&self.data[start..=stop]);
    }
}

impl std::ops::Index<[isize; 2]> for SphericalHarmonicsArray {
    type Output = f64;
    fn index(&self, index: [isize; 2]) -> &f64 {
        &self.data[self.linear_index(index)]
    }
}

impl std::ops::IndexMut<[isize; 2]> for SphericalHarmonicsArray {
    fn index_mut(&mut self, index: [isize; 2]) -> &mut f64 {
        let i = self.linear_index(index);
        &mut self.data[i]
    }
}

impl std::fmt::Debug for SphericalHarmonicsArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SphericalHarmonicsArray[\n  l \\ m  ")?;
        for m in -self.max_angular..=self.max_angular {
            write!(f, " {: ^12}", m)?;
        }
        writeln!(f)?;
        for l in 0..(self.max_angular + 1) {
            write!(f, "  {: <8}", l)?;
            for _ in -self.max_angular..-l {
                write!(f, "             ")?;
            }

            for m in -l..=l {
                write!(f, " {:+.9}", self[[l, m]])?;
            }
            writeln!(f)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Compute a full set of spherical harmonics at given positions
///
/// Follows the algorithm described in <https://arxiv.org/abs/1410.1748>
#[derive(Debug, Clone)]
pub struct SphericalHarmonics {
    max_angular: usize,
    /// array of associated Legendre polynomials
    legendre_polynomials: LegendreArray,
    /// 'A' coefficient from the arxiv paper to compute Legendre polynomials
    coefficient_a: LegendreArray,
    /// 'B' coefficient from the arxiv paper to compute Legendre polynomials
    coefficient_b: LegendreArray,
    /// used for gradients, `sqrt((l + m) * (l - m + 1)) * L_l^{m - 1} - sqrt((l - m) * (l + m + 1)) * P_l^{m + 1}`
    delta_legendre_polynomials: LegendreArray,
    /// used for gradients, either `m / sin(θ) P_l^m` or `- 1 / (2 cos(θ)) *
    /// ∆P_l^m` depending on the value of theta. This shifts the singularity
    /// coming from `1 / sin(θ)` from the poles to the equator so that we never
    /// have to deal with it.
    legendre_over_theta: LegendreArray,
}

impl SphericalHarmonics {
    /// Build a new `SphericalHarmonics` calculator with the given `l_max`, and
    /// pre-compute all required quantities
    pub fn new(max_angular: usize) -> SphericalHarmonics {
        let mut coefficient_a = LegendreArray::new(max_angular);
        let mut coefficient_b = LegendreArray::new(max_angular);
        for l in 2..(max_angular + 1) {
            let ls = (l * l) as f64;
            let lm1s = ((l as isize - 1) * (l as isize - 1)) as f64;
            for m in 0..(l - 1) {
                let ms = (m * m) as f64;
                coefficient_a[[l, m]] = f64::sqrt((4.0 * ls - 1.0) / (ls - ms));
                coefficient_b[[l, m]] = -f64::sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
            }
        }

        SphericalHarmonics {
            max_angular: max_angular,
            legendre_polynomials: LegendreArray::new(max_angular),
            delta_legendre_polynomials: LegendreArray::new(max_angular),
            legendre_over_theta: LegendreArray::new(max_angular),
            coefficient_a: coefficient_a,
            coefficient_b: coefficient_b,
        }
    }

    /// Evaluate the Legendre polynomials at `cos(θ)`, and fill
    /// `self.legendre_polynomials` with the resulting values
    fn compute_legendre_polynomials(&mut self, cos_theta: f64, sin_theta: f64) {
        let mut value = SQRT_1_OVER_2PI;
        self.legendre_polynomials[[0, 0]] = value;

        if self.max_angular > 0 {
            self.legendre_polynomials[[1, 0]] = cos_theta * SQRT_3 * value;
            value *= -SQRT_3_OVER_2 * sin_theta;
            self.legendre_polynomials[[1, 1]] = value;

            let a = &mut self.coefficient_a;
            let b = &mut self.coefficient_b;
            let p = &mut self.legendre_polynomials;

            for l in 2..(self.max_angular + 1) {
                for m in 0..(l - 1) {
                    p[[l, m]] = a[[l, m]] * (cos_theta * p[[l - 1, m]] + b[[l, m]] * p[[l - 2, m]]);
                }

                p[[l, l - 1]] = cos_theta * f64::sqrt(2.0 * l as f64 + 1.0) * value;
                value *= -f64::sqrt(1.0 + 0.5 / l as f64) * sin_theta;
                p[[l, l]] = value;
            }
        }
    }

    /// Compute factors required for the derivatives of spherical harmonics at
    /// `cos(θ)`, and fill `self.delta_legendre_polynomials` and
    /// `self.legendre_over_theta` with the values.
    fn compute_derivative_factors(&mut self, cos_theta: f64, sin_theta: f64) {
        let compute_delta_legendre = |l, m, p_m_l_minus_1, p_m_l_plus_1| {
            f64::sqrt(((l + m) * (l - m + 1)) as f64) * p_m_l_minus_1
            - f64::sqrt(((l - m) * (l + m + 1)) as f64) * p_m_l_plus_1
        };

        self.delta_legendre_polynomials[[0, 0]] = 0.0;

        for l in 1..(self.max_angular + 1) {
            let m = 0;
            // from P_l^{−m} = (−1)^m (l − m)!/(l + m)! P_l^m
            let p_m_l_minus_1 = -1.0 / ((l * l + l) as f64) * self.legendre_polynomials[[l, 1]];
            let p_m_l_plus_1 = self.legendre_polynomials[[l, m + 1]];
            self.delta_legendre_polynomials[[l, m]] = compute_delta_legendre(l, m, p_m_l_minus_1, p_m_l_plus_1);

            for m in 1..l {
                let p_m_l_minus_1 = self.legendre_polynomials[[l, m - 1]];
                let p_m_l_plus_1 = self.legendre_polynomials[[l, m + 1]];

                self.delta_legendre_polynomials[[l, m]] = compute_delta_legendre(l, m, p_m_l_minus_1, p_m_l_plus_1);
            }

            let m = l;
            let p_m_l_minus_1 = self.legendre_polynomials[[l, m - 1]];
            let p_m_l_plus_1 = 0.0;
            self.delta_legendre_polynomials[[l, m]] = compute_delta_legendre(l, m, p_m_l_minus_1, p_m_l_plus_1);
        }

        // legendre_over_theta
        if sin_theta > 0.1 {
            for l in 0..(self.max_angular + 1) {
                for m in 0..=l {
                    self.legendre_over_theta[[l, m]] = m as f64 / sin_theta * self.legendre_polynomials[[l, m]];
                }
            }
        } else {
            for l in 0..(self.max_angular + 1) {
                for m in 0..=l {
                    self.legendre_over_theta[[l, m]] = -0.5 / cos_theta * self.delta_legendre_polynomials[[l, m]];
                }
            }
        }
    }

    /// Evaluate all spherical harmonics for the given `direction`, and store
    /// the results in `values`. If `gradients` is `Some`, then this function
    /// also computes cartesian gradients and store them in `gradients`.
    #[time_graph::instrument(name = "SphericalHarmonics::compute")]
    pub fn compute(
        &mut self,
        direction: Vector3D,
        values: &mut SphericalHarmonicsArray,
        mut gradients: Option<&mut [SphericalHarmonicsArray; 3]>
    ) {
        assert!(
            (direction.norm2() - 1.0).abs() < 1e-9,
            "expected the direction vector to be normalized in spherical harmonics"
        );
        assert_eq!(
            values.max_angular as usize, self.max_angular,
            "wrong size for the values array, expected max_angular to be {}, got {}",
            self.max_angular, values.max_angular,
        );
        if let Some(ref gradients) = gradients {
            for i in 0..3 {
                assert_eq!(
                    gradients[i].max_angular as usize, self.max_angular,
                    "wrong size for one gradient array, expected max_angular to be {}, got {}",
                    self.max_angular, gradients[i].max_angular,
                );
            }
        }

        let sqrt_xy = f64::hypot(direction[0], direction[1]);
        let cos_theta = direction[2];
        let sin_theta = sqrt_xy;

        let (cos_phi, sin_phi) = if sqrt_xy > f64::EPSILON {
            (direction[0] / sqrt_xy, direction[1] / sqrt_xy)
        } else {
            (1.0, 0.0)
        };

        self.compute_legendre_polynomials(cos_theta, sin_theta);
        if gradients.is_some() {
            self.compute_derivative_factors(cos_theta, sin_theta);
        }

        for l in 0..(self.max_angular + 1) {
            // compute values for m = 0 first
            values[[l as isize, 0]] = self.legendre_polynomials[[l, 0]] / SQRT_2;
        }

        if let Some(ref mut gradients) = gradients {
            // gradients for m = 0
            gradients[0][[0, 0]] = 0.0;
            gradients[1][[0, 0]] = 0.0;
            gradients[2][[0, 0]] = 0.0;
            for l in 1..(self.max_angular + 1) {
                let legendre_factor = f64::sqrt(0.5 * (l * (l + 1)) as f64) * self.legendre_polynomials[[l, 1]];

                // d/dx: cos(ϕ) cos(θ) sqrt(l * (l + 1) / 2) * P_l^1(cos(θ))
                gradients[0][[l as isize, 0]] = cos_phi * cos_theta * legendre_factor;
                // d/dy: sin(ϕ) cos(θ) sqrt(l * (l + 1) / 2) * P_l^1(cos(θ))
                gradients[1][[l as isize, 0]] = sin_phi * cos_theta * legendre_factor;
                // d/dz: -sin(θ) sqrt(l * (l + 1) / 2) * P_l^1(cos(θ))
                gradients[2][[l as isize, 0]] = -sin_theta * legendre_factor;
            }
        }

        // Use a recurrence relation for sin(m ϕ) and cos(m ϕ) for m ≠ 0. The
        // relation used differ partially from the arxiv paper to follow the
        // convention documented on Wikipedia for real spherical harmonics
        // (https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form). This
        // effectively cancels out the Condon-Shortley phase (-1^m) in the final
        // real spherical harmonics.

        // initialize recurrence for m=-1
        let mut cos_1 = 1.0;
        let mut sin_1 = 0.0;
        // initialize recurrence for m=0
        let mut cos_2 = -cos_phi;     // <== this was changed from `cos` to `-cos`
        let mut sin_2 = sin_phi;      // <== this was changed from `-sin` to `sin`

        let minus_two_cos = -2.0 * cos_phi; // <== this was changed from `2 cos` to `-2 cos`
        for m in 1..(self.max_angular + 1) {
            let sin_m_phi = minus_two_cos * sin_1 - sin_2;
            let cos_m_phi = minus_two_cos * cos_1 - cos_2;
            sin_2 = sin_1;
            sin_1 = sin_m_phi;
            cos_2 = cos_1;
            cos_1 = cos_m_phi;

            for l in m..(self.max_angular + 1) {
                let p_lm = self.legendre_polynomials[[l, m]];
                values[[l as isize, m as isize]] = p_lm * cos_m_phi;
                values[[l as isize, -(m as isize)]] = p_lm * sin_m_phi;
            }

            if let Some(ref mut gradients) = gradients {
                // gradients for m ≠ 0
                for l in m..(self.max_angular + 1) {
                    // ∆P_l^m = sqrt((l + m) * (l - m + 1)) * L_l^{m - 1} - sqrt((l - m) * (l + m + 1)) * P_l^{m + 1}
                    let delta_p_lm = self.delta_legendre_polynomials[[l, m]];

                    let sin_m_phi_delta_p_lm = sin_m_phi * delta_p_lm;
                    let cos_m_phi_delta_p_lm = cos_m_phi * delta_p_lm;

                    // m / sin(θ) * P_l^m
                    let p_lm_over_theta = self.legendre_over_theta[[l, m]];

                    // m>0, d/dx: m sin(ϕ) / sin(θ) * sin(m ϕ) P_l^m - cos(θ) cos(ϕ) / 2 * cos(m ϕ) ∆P_l^m
                    gradients[0][[l as isize, m as isize]] = sin_phi * p_lm_over_theta * sin_m_phi - 0.5 * cos_theta * cos_phi * cos_m_phi_delta_p_lm;
                    // m<0, d/dx: -m sin(ϕ)/sin(θ) * cos(m ϕ) P_l^m - cos(θ) cos(ϕ) / 2 * sin(m ϕ) ∆P_l^m
                    gradients[0][[l as isize, -(m as isize)]] = -sin_phi * p_lm_over_theta * cos_m_phi - 0.5 * cos_theta * cos_phi * sin_m_phi_delta_p_lm;

                    // m>0, d/dy: - m cos(ϕ) / sin(θ) * sin(m ϕ) P_l^m - cos(θ) sin(ϕ) / 2 * cos(m ϕ) ∆P_l^m
                    gradients[1][[l as isize, m as isize]] = - cos_phi * p_lm_over_theta * sin_m_phi - 0.5 * cos_theta * sin_phi * cos_m_phi_delta_p_lm;
                    // m<0, d/dy: m cos(ϕ) / sin(θ) * cos(m ϕ) P_l^m - cos(θ) sin(ϕ) / 2 * sin(m ϕ) ∆P_l^m
                    gradients[1][[l as isize, -(m as isize)]] = cos_phi * p_lm_over_theta * cos_m_phi - 0.5 * cos_theta * sin_phi * sin_m_phi_delta_p_lm;

                    // m>0, d/dz: sin(θ) / 2 * cos(m ϕ) ∆P_l^m
                    gradients[2][[l as isize, m as isize]] = 0.5 * sin_theta * cos_m_phi_delta_p_lm;
                    // m<0, d/dz: sin(θ) / 2 * sin(m ϕ) ∆P_l^m
                    gradients[2][[l as isize, -(m as isize)]] = 0.5 * sin_theta * sin_m_phi_delta_p_lm;
                }
            }
        }
    }
}


/// Store together the spherical harmonics implementation and cached allocation
/// for values/gradients.
pub(crate) struct SphericalHarmonicsCache {
    /// Implementation of the spherical harmonics
    code: SphericalHarmonics,
    /// Cache for the spherical harmonics values
    pub(crate) values: SphericalHarmonicsArray,
    /// Cache for the spherical harmonics gradients (one value each for x/y/z)
    pub(crate) gradients: [SphericalHarmonicsArray; 3],
}

impl SphericalHarmonicsCache {
    /// Create a new `SphericalHarmonicsCache` for the given `max_angular` parameter
    pub(crate) fn new(max_angular: usize) -> SphericalHarmonicsCache {
        let code = SphericalHarmonics::new(max_angular);
        let values = SphericalHarmonicsArray::new(max_angular);
        let gradients = [
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular)
        ];

        return SphericalHarmonicsCache { code, values, gradients };
    }

    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub(crate) fn compute(&mut self, direction: Vector3D, gradient: bool) {
        if gradient {
            self.code.compute(
                direction,
                &mut self.values,
                Some(&mut self.gradients),
            );
        } else {
            self.code.compute(
                direction,
                &mut self.values,
                None,
            );
        }

    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use approx::assert_relative_eq;
    use super::*;

    #[test]
    fn linear_index_legendre_array() {
        // ensure each 2D index is mapped to a separate linear index
        let max_angular = 20;

        let mut count = 0;
        let mut set = HashSet::new();
        let array = LegendreArray::new(max_angular);
        for l in 0..(max_angular + 1) {
            for m in 0..=l {
                set.insert(array.linear_index([l, m]));
                count += 1;
            }
        }

        assert_eq!(count, set.len());
        assert_eq!(array.data.len(), set.len());
    }

    #[test]
    fn linear_index_sph_array() {
        // ensure each 2D index is mapped to a separate linear index
        let max_angular = 20;

        let mut count = 0;
        let mut set = HashSet::new();
        let array = SphericalHarmonicsArray::new(max_angular);
        for l in 0..(max_angular as isize + 1) {
            for m in -l..=l {
                set.insert(array.linear_index([l, m]));
                count += 1;
            }
        }

        assert_eq!(count, set.len());
        assert_eq!(array.data.len(), set.len());
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn finite_differences() {
        let mut directions = [
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.0, 0.0, 1.0),
            Vector3D::new(1.0, 1.0, 1.0),
            Vector3D::new(1.0, -3.0, 9.0),
            Vector3D::new(1.0, 8.0, 12.0),
            Vector3D::new(-452.0, 825.0, 22.0),
        ];

        for d in &mut directions {
            *d /= d.norm();
        }

        let max_angular = 25;
        let mut spherical_harmonics = SphericalHarmonics::new(max_angular);
        let mut values = SphericalHarmonicsArray::new(max_angular);
        let mut values_delta = SphericalHarmonicsArray::new(max_angular);
        let mut gradients = [
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular),
            SphericalHarmonicsArray::new(max_angular)
        ];

        let delta = 1e-9;
        for &rij in &directions {
            spherical_harmonics.compute(rij, &mut values, Some(&mut gradients));

            let mut rij_dx = rij + Vector3D::new(delta, 0.0, 0.0);
            rij_dx /= rij_dx.norm();
            spherical_harmonics.compute(rij_dx, &mut values_delta, None);
            for l in 0..(max_angular as isize + 1) {
                for m in -l..(l + 1) {
                    let finite_difference_x = (values_delta[[l, m]] - values[[l, m]]) / delta;
                    assert_relative_eq!(
                        finite_difference_x, gradients[0][[l, m]],
                        epsilon=1e-5, max_relative=1e-5
                    );
                }
            }

            let mut rij_dy = rij + Vector3D::new(0.0, delta, 0.0);
            rij_dy /= rij_dy.norm();
            spherical_harmonics.compute(rij_dy, &mut values_delta, None);
            for l in 0..(max_angular as isize + 1) {
                for m in -l..(l + 1) {
                    let finite_difference_y = (values_delta[[l, m]] - values[[l, m]]) / delta;
                    assert_relative_eq!(
                        finite_difference_y, gradients[1][[l, m]],
                        epsilon=1e-5, max_relative=1e-5
                    );
                }
            }

            let mut rij_dz = rij + Vector3D::new(0.0, 0.0, delta);
            rij_dz /= rij_dz.norm();
            spherical_harmonics.compute(rij_dz, &mut values_delta, None);

            for l in 0..(max_angular as isize + 1) {
                for m in -l..(l + 1) {
                    let finite_difference_z = (values_delta[[l, m]] - values[[l, m]]) / delta;
                    assert_relative_eq!(
                        finite_difference_z, gradients[2][[l, m]],
                        epsilon=1e-5, max_relative=1e-5
                    );
                }
            }
        }
    }

    mod bad {
        use super::super::{SphericalHarmonics, SphericalHarmonicsArray};
        use crate::Vector3D;

        #[test]
        #[should_panic = "wrong size for the values array, expected max_angular to be 3, got 5"]
        fn value_array_size() {
            let mut spherical_harmonics = SphericalHarmonics::new(3);
            let mut values = SphericalHarmonicsArray::new(5);

            spherical_harmonics.compute(Vector3D::new(1.0, 0.0, 0.0), &mut values, None);
        }

        #[test]
        #[should_panic = "wrong size for one gradient array, expected max_angular to be 3, got 5"]
        fn gradient_array_size() {
            let mut spherical_harmonics = SphericalHarmonics::new(3);
            let mut values = SphericalHarmonicsArray::new(3);
            let mut gradients = [
                SphericalHarmonicsArray::new(5),
                SphericalHarmonicsArray::new(5),
                SphericalHarmonicsArray::new(5),
            ];

            spherical_harmonics.compute(Vector3D::new(1.0, 0.0, 0.0), &mut values, Some(&mut gradients));
        }

        #[test]
        #[should_panic = "expected the direction vector to be normalized"]
        fn non_normalized_direction() {
            let mut spherical_harmonics = SphericalHarmonics::new(3);
            let mut values = SphericalHarmonicsArray::new(3);

            spherical_harmonics.compute(Vector3D::new(1.0, 1.0, 1.0), &mut values, None);
        }
    }
}
