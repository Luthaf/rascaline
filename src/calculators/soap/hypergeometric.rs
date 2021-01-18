//! This module implement a modified version of Kummer's confluent
//! hypergeometric function 1F1: `G(rij) = Γ(a) / Γ(b) exp(-c rij^2) 1F1(a, b, c^2
//! rij^2 / (c + d)` where `a = (n + l + 3)/2`, `b = l + 3/2`, `c = 1 / 2σ^2`,
//! `d = 1 / 2 (r_cut \sqrt(n) / (n_max + 1))^2`; σ is the atomic density
//! gaussian width, `r_cut` the cutoff radius, `n_max` the number of radial
//! basis, n the current radial basis index and l the current angular index.
//!
//! This function is used to compute the radial integral in SOAP spherical
//! expansion when using GTO radial basis.
//!
//! This code was originally written by Félix Musil @ COSMO/EPFL

use ndarray::{Array2, ArrayViewMut2, Axis, azip};
use log::warn;

use crate::math::gamma;

// The precision up to which we converge our implementation of the G function
const HYPERGEOMETRIC_PRECISION: f64 = 1e-13;

/// Compute 2F0 as a the corresponding series truncated to the first n terms.
#[derive(Debug, Clone)]
struct Series2F0 {
    coefficients: Vec<f64>,
}

impl Series2F0 {
    /// Create a new `Series2F0` computing the function itself
    pub fn new(a: f64, b: f64, n_terms: usize) -> Series2F0 {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        let mut coefficients = Vec::new();
        coefficients.reserve(n_terms);

        let mut coefficient = (b - a) * (1.0 - a);
        coefficients.push(coefficient);
        for i in (1..n_terms).map(|i| i as f64) {
            coefficient *= (b - a + i) * (1.0 - a + i) / (i + 1.0);
            coefficients.push(coefficient);
        }

        return Series2F0 {
            coefficients: coefficients,
        }
    }

    /// Create a new `Series2F0` computing the function derivative
    pub fn derivative(a: f64, b: f64, n_terms: usize) -> Series2F0 {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        let mut coefficients = Vec::new();
        coefficients.reserve(n_terms);

        let mut coefficient = (b - a) * -a;
        coefficients.push(coefficient);
        for i in (1..n_terms).map(|i| i as f64) {
            coefficient *= (b - a + i) * (i - a) / (i + 1.0);
            coefficients.push(coefficient);
        }

        return Series2F0 {
            coefficients: coefficients,
        }
    }

    /// Actually compute the function for input parameter `z`
    pub fn compute(&self, z: f64) -> f64 {
        let mut result = 1.0;
        let iz = 1.0 / z;
        let mut iz_pow = 1.0;
        for coefficient in &self.coefficients {
            iz_pow *= iz;
            result += coefficient * iz_pow;
        }
        return result;
    }
}

/// Compute 1F1 as a the corresponding series truncated to the first n terms.
#[derive(Debug, Clone)]
struct Series1F1 {
    coefficients: Vec<f64>,
}

impl Series1F1 {
    /// Create a new `Series1F1` computing the function itself
    pub fn new(a: f64, b: f64, n_terms: usize) -> Series1F1 {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        let mut coefficients = Vec::new();
        coefficients.reserve(n_terms);

        let mut coefficient = a / b;
        coefficients.push(coefficient);
        for i in (1..n_terms).map(|i| i as f64) {
            coefficient *= (a + i) / ((i + 1.0) * (b + i));
            coefficients.push(coefficient);
        }

        return Series1F1 {
            coefficients: coefficients,
        }
    }

    /// Create a new `Series1F1` computing the function derivative
    pub fn derivative(a: f64, b: f64, n_terms: usize) -> Series1F1 {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        let mut coefficients = Vec::new();
        coefficients.reserve(n_terms);

        let mut coefficient = (a + 1.0) / (b + 1.0);
        coefficients.push(coefficient);
        for i in (1..n_terms).map(|i| i as f64) {
            coefficient *= (a + i + 1.0) / ((i + 1.0) * (b + i + 1.0));
            coefficients.push(coefficient);
        }

        return Series1F1 {
            coefficients: coefficients,
        }
    }

    /// Actually compute the function for input parameter `z`
    #[allow(clippy::identity_op)]
    pub fn compute(&self, z: f64) -> f64 {
        let mut result = 1.0;
        let mut z_pow = z;
        let z4 = z * z * z * z;
        // Use an adaptive summation: computes several terms at a time and check
        // if required accuracy have been reached (typical n. of terms needed is
        // ~20)
        for i in (0..self.coefficients.len()).step_by(4)  {
            let mut term = self.coefficients[i + 3];
            term = self.coefficients[i + 2] + z * term;
            term = self.coefficients[i + 1] + z * term;
            term = self.coefficients[i + 0] + z * term;
            term *= z_pow;

            if term < HYPERGEOMETRIC_PRECISION * result {
                result += term;
                break;
            }

            z_pow *= z4;
            result += term;
        }

        return result;
    }
}

/// Compute G using the direct sum for 1F1
///
/// `1F1(a, b, z) = \sum_{j=0}^{\infty} \frac{(a)_j}{(b)_jj!} z^{j}`
#[derive(Debug, Clone)]
struct HyperGeometricSeries {
    /// First parameter to the 1F1 function
    a: f64,
    /// Second parameter to the 1F1 function
    b: f64,
    /// Store precomputed `\frac{\Gamma{a}}{\Gamma{b}}`
    gamma_ratio: f64,
    /// Actual implementation of 1F1, computing the function
    series_1f1: Series1F1,
    /// Actual implementation of 1F1, computing the function derivative
    series_1f1_derivative: Series1F1,
}

impl HyperGeometricSeries {
    /// Create a new `HyperGeometricSeries` for the given a and b parameters
    pub fn new(a: f64, b: f64) -> HyperGeometricSeries {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        HyperGeometricSeries {
            a,
            b,
            gamma_ratio: gamma(a) / gamma(b),
            series_1f1: Series1F1::new(a, b, 200),
            series_1f1_derivative: Series1F1::derivative(a, b, 200),
        }
    }

    /// Computes G(z), with `z = c^2 rij^2 / (c + d)` and `z2 = -c rij^2`
    ///
    /// # Warning
    ///
    /// The derivative this function computes is not dG/dz but `d1F1/dz *
    /// \frac{\Gamma(a)}{\Gamma(b)} * \exp{-\alpha r_{ij}^2}`. We do this to
    /// avoid computing both d1F1/dz and 1F1 when asking for gradients and
    /// perform this step in `HyperGeometricSphericalExpansion`.
    pub fn compute(&self, z: f64, z2: f64, derivative: bool) -> f64 {
        if derivative {
            self.gamma_ratio * self.series_1f1_derivative.compute(z) * f64::exp(z2) * self.a / self.b
        } else {
            self.gamma_ratio * self.series_1f1.compute(z) * f64::exp(z2)
        }
    }

    /// Compute 1F1 itself to check for the best validity domain of this
    /// implementation
    pub fn compute_1f1(&self, z: f64) -> f64 {
        self.series_1f1.compute(z)
    }
}

/// Compute G using the asymptotic expansion of 1F1 as the third argument goes
/// to infinity:
///
/// `1F1(a,b,z) \sim \exp{z} z^{a-b} \frac{\Gamma{b}}{\Gamma{a}} 2F0(a, b, z)`
///
/// where `2F0(a, b, z) = \sum_{j=0}^{\infty} \frac{(b-a)_j(1-a)_j}{j!} z^{-j}`
#[derive(Debug, Clone)]
struct HyperGeometricAsymptotic {
    /// First parameter to the 1F1 function
    a: f64,
    /// Second parameter to the 1F1 function
    b: f64,
    /// Store precomputed `\frac{\Gamma{a}}{\Gamma{b}}`
    gamma_ratio: f64,
    /// Actual implementation of 2F0, computing the function
    series_2f0: Series2F0,
    /// Actual implementation of 2F0, computing the function derivative
    series_2f0_derivative: Series2F0,
}

impl HyperGeometricAsymptotic {
    /// Create a new `HyperGeometricAsymptotic` for the given a and b parameters
    pub fn new(a: f64, b: f64) -> HyperGeometricAsymptotic {
        assert!(a > 0.0 && b > 0.0, "a and b must be positive");

        HyperGeometricAsymptotic {
            a,
            b,
            gamma_ratio: gamma(a) / gamma(b),
            series_2f0: Series2F0::new(a, b, 20),
            series_2f0_derivative: Series2F0::derivative(a, b, 20),
        }
    }

    /// Computes G(z), with `z = c^2 rij^2 / (c + d)` and `z2 = -c rij^2`
    ///
    /// # Warning
    ///
    /// The derivative this function computes is not dG/dz but `d1F1/dz *
    /// \frac{\Gamma(a)}{\Gamma(b)} * \exp{-\alpha r_{ij}^2}`. We do this to
    /// avoid computing both d1F1/dz and 1F1 when asking for gradients and
    /// perform this step in `HyperGeometricSphericalExpansion`.
    pub fn compute(&self, z: f64, z2: f64, derivative: bool) -> f64 {
        let hyp2f0 = if derivative {
            self.series_2f0_derivative.compute(z)
        } else {
            self.series_2f0.compute(z)
        };
        let factor = z.powf(self.a - self.b);
        return hyp2f0 * f64::exp(z + z2) * factor;
    }

    /// Compute 1F1 itself to check for the best validity domain of this
    /// implementation
    pub fn compute_1f1(&self, z: f64) -> f64 {
        let factor = z.powf(self.a - self.b);
        return f64::exp(z) * factor * self.series_2f0.compute(z) / self.gamma_ratio;
    }
}

/// Switch between the asymptotic expansion or the series implementations of our
/// hypergeometric function G(z) depending on the arguments domains.
#[derive(Debug, Clone)]
pub struct HyperGeometric {
    /// when a == b, 1F1 is an exponential, so we use a fast path in this case
    is_exponential: bool,
    /// for the exponential case, store Γ(a) / Γ(b)
    gamma_ratio: f64,
    /// Asymptotic implementation of the G function
    asymptotic: HyperGeometricAsymptotic,
    /// Series implementation of the G function
    series: HyperGeometricSeries,
    /// For z above this value, use the asymptotic implementation, for z below
    /// this value use the series implementation
    switching_point: f64,
}

impl HyperGeometric {
    pub fn new(a: f64, b: f64) -> HyperGeometric {
        // We try to determine what is the switching point between power series
        // and asymptotic expansion. We choose the method that requires fewer
        // terms for a given target accuracy. The asymptotic expansion tends to
        // overflow at the switching point.

        const MAX_ITERATIONS: usize = 100;

        let series = HyperGeometricSeries::new(a, b);
        let asymptotic = HyperGeometricAsymptotic::new(a, b);

        if f64::abs(a - b) < 100.0 * f64::EPSILON {
            return HyperGeometric {
                is_exponential: true,
                gamma_ratio: gamma(a) / gamma(b),
                series: series,
                asymptotic: asymptotic,
                switching_point: 0.0,
            }
        }

        // Find the largest z for which the 2 definitions agree within tolerance
        // using bisection.

        // brackets the switching point
        let mut z_switch = 1.0;
        let mut z_below  = 1.0;
        let mut z_above = 1.0;

        let precision = 1e2 * HYPERGEOMETRIC_PRECISION;
        if f64::abs(1.0 - series.compute_1f1(z_switch) / asymptotic.compute_1f1(z_switch)) > HYPERGEOMETRIC_PRECISION {
            for _ in 0..MAX_ITERATIONS {
                z_above *= 1.5;
                let s = series.compute_1f1(z_above);
                let a = asymptotic.compute_1f1(z_above);
                if f64::abs(1.0 - s / a) < precision {
                    break;
                }
            }
        } else {
            for _ in 0..MAX_ITERATIONS {
                z_below *= 0.5;
                let s = series.compute_1f1(z_below);
                let a = asymptotic.compute_1f1(z_below);

                if f64::abs(1.0 - s / a) > precision {
                    break;
                }
            }
        }

        // and now bisects until we are reasonably close to an accurate
        // determination
        let mut accuracy_reached = false;
        z_switch = (z_above + z_below) * 0.5;
        let mut series_result = series.compute_1f1(z_switch);
        let mut asymptotic_result = asymptotic.compute_1f1(z_switch);
        for _ in 0..MAX_ITERATIONS {
            if f64::abs(1.0 - series_result / asymptotic_result) > HYPERGEOMETRIC_PRECISION {
                z_below = z_switch;
            } else {
                z_above = z_switch;
            }

            z_switch = (z_above + z_below) * 0.5;
            series_result = series.compute_1f1(z_switch);
            asymptotic_result = asymptotic.compute_1f1(z_switch);

            if z_above - z_below < HYPERGEOMETRIC_PRECISION {
                accuracy_reached = true;
                break;
            }
        }

        if !accuracy_reached {
            warn!(
                "failed to reach sufficient accuracy for the \
                hypergeometric function with parameters a={} and b={}",
                a, b
            );
        }

        HyperGeometric {
            is_exponential: false,
            gamma_ratio: 0.0,
            series: series,
            asymptotic: asymptotic,
            switching_point: z_switch,
        }
    }

    pub fn compute(&self, z: f64, z2: f64, derivative: bool) -> f64 {
        let result;
        if self.is_exponential {
            result = self.gamma_ratio * f64::exp(z + z2);
        } else if z > self.switching_point {
            result = self.asymptotic.compute(z, z2, derivative);
        } else {
            result = self.series.compute(z, z2, derivative);
        }

        debug_assert!(
            result.is_finite(),
            "HyperGeometric overflowed with z={}", z
        );

        return result;
    }

    /// Compute 1F1 itself, used in tests to check that this part of the
    /// implementation is correct
    #[cfg(test)]
    fn compute_1f1(&self, z: f64) -> f64 {
        let result;
        if self.is_exponential {
            result = f64::exp(z);
        } else if z > self.switching_point {
            result = self.asymptotic.compute_1f1(z);
        } else {
            result = self.series.compute_1f1(z);
        }

        debug_assert!(
            result.is_finite(),
            "HyperGeometric overflowed with z={}", z
        );

        return result;
    }
}

/// Computes the G function and its derivative for all possible values of `l <
/// l_max + 1` and `n < n_max` with `a = (n + l + 3) / 2` and b = `l + 3/2`
/// using recurrence relationships between different n/l values to speedup the
/// full calculation.
///
/// `G(a, b, z) = \frac{\Gamma(a)}{\Gamma(b)} * \exp{-\alpha r_{ij}^2} 1F1(a, b,
/// z)`
#[derive(Debug, Clone)]
pub struct HyperGeometricSphericalExpansion {
    /// n_max parameter
    max_radial: usize,
    /// l_max parameter
    max_angular: usize,
    /// HyperGeometric implementation in a n_max x l_max + 1 array
    hypergeometric: Array2<HyperGeometric>,
}

#[derive(Debug, Clone, Copy)]
pub struct HyperGeometricParameters<'a> {
    pub atomic_gaussian_constant: f64,
    pub gto_gaussian_constants: &'a [f64],
}

impl HyperGeometricSphericalExpansion {
    pub fn new(max_radial: usize, max_angular: usize) -> Self {

        let mut hypergeometric = Vec::new();
        hypergeometric.reserve(max_angular * (max_radial + 1));

        for n in 0..max_radial {
            for l in 0..(max_angular + 1) {
                let a = 0.5 * (n + l + 3) as f64;
                let b = l as f64 + 1.5;
                hypergeometric.push(HyperGeometric::new(a, b));
            }
        }

        let hypergeometric = Array2::from_shape_vec(
            (max_radial, max_angular + 1), hypergeometric
        ).expect("wrong shape in Array2::from_shape_vec");
        return HyperGeometricSphericalExpansion {
            max_angular: max_angular,
            max_radial: max_radial,
            hypergeometric: hypergeometric,
        }
    }

    #[allow(clippy::collapsible_if)]
    pub fn compute(
        &self,
        rij: f64,
        parameters: HyperGeometricParameters,
        values: ArrayViewMut2<f64>,
        gradients: Option<ArrayViewMut2<f64>>,
    ) {
        assert_eq!(parameters.gto_gaussian_constants.len(), self.max_radial);
        assert_eq!(values.shape(), [self.max_radial, self.max_angular + 1]);
        if let Some(ref gradients) = gradients {
            assert_eq!(gradients.shape(), [self.max_radial, self.max_angular + 1]);
        }

        if self.max_angular < 3 {
            // recursion needs 4 evaluations of 1F1 so not worth it if l_max < 3
            self.direct(rij, parameters, values, gradients);
        } else {
            self.recursive(rij, parameters, values, gradients);
        }
    }

    fn direct(
        &self,
        rij: f64,
        parameters: HyperGeometricParameters,
        mut values: ArrayViewMut2<f64>,
        mut gradients: Option<ArrayViewMut2<f64>>,
    ) {

        let alpha = parameters.atomic_gaussian_constant;
        let alpha_rij = alpha * rij;
        let z2 = -rij * alpha_rij;

        for n in 0..self.max_radial {
            let z = alpha_rij * alpha_rij / (alpha + parameters.gto_gaussian_constants[n]);
            for l in 0..(self.max_angular + 1) {
                values[[n, l]] = self.hypergeometric[[n, l]].compute(z, z2, false);
            }
        }

        if let Some(ref mut gradients) = gradients {
            for n in 0..self.max_radial {
                let z = alpha_rij * alpha_rij / (alpha + parameters.gto_gaussian_constants[n]);

                for l in 0..(self.max_angular + 1) {
                    gradients[[n, l]] = self.hypergeometric[[n, l]].compute(z, z2, true);
                }
                let mut row = gradients.index_axis_mut(Axis(0), n);
                row *= 2.0 * z / rij;
            }

            azip!((gradient in gradients, &value in &values)
                *gradient -= 2.0 * alpha * rij * value
            );
        }
    }

    #[allow(clippy::similar_names)]
    fn recursive(
        &self,
        rij: f64,
        parameters: HyperGeometricParameters,
        mut values: ArrayViewMut2<f64>,
        mut gradients: Option<ArrayViewMut2<f64>>,
    ) {
        assert!(self.max_angular >= 3);

        let alpha = parameters.atomic_gaussian_constant;
        let alpha_rij = alpha * rij;
        let z2 = -alpha_rij * rij;

        let get_ab = |n, l| (0.5 * (n + l + 3) as f64, l as f64 + 1.5);

        for n in 0..self.max_radial {
            // get the starting points for the recursion
            let z = alpha_rij * alpha_rij / (alpha + parameters.gto_gaussian_constants[n]);

            let l = self.max_angular;
            let mut m1p2p = self.hypergeometric[[n, l]].compute(z, z2, false);
            values[[n, l]] = m1p2p;

            let mut m2p3p = self.hypergeometric[[n, l]].compute(z, z2, true);
            if let Some(ref mut gradients) = gradients {
                gradients[[n, l]] = m2p3p;
            }

            let l = self.max_angular - 1;
            let mut mp1p2p = self.hypergeometric[[n, l]].compute(z, z2, false);
            values[[n, l]] = mp1p2p;

            let mut mp2p3p = self.hypergeometric[[n, l]].compute(z, z2, true);
            if let Some(ref mut gradients) = gradients {
                gradients[[n, l]] = mp2p3p;
            }

            let mut l = self.max_angular;
            while l > 2 {
                l -= 2;

                let (a, b) = get_ab(n, l);
                let m1p1p = g_gradient_recursive_step(a, b, z, m2p3p, m1p2p);
                let m00 = g_value_recursive_step(a, b, z, m1p2p, m1p1p);

                values[[n, l]] = m00;
                if let Some(ref mut gradients) = gradients {
                    gradients[[n, l]] = m1p1p;
                }

                m2p3p = m1p1p;
                m1p2p = m00;

                let (a, b) = get_ab(n, l - 1);
                let mp1p1p = g_gradient_recursive_step(a, b, z, mp2p3p, mp1p2p);
                let mp00 = g_value_recursive_step(a, b, z, mp1p2p, mp1p1p);

                values[[n, l - 1]] = mp00;
                if let Some(ref mut gradients) = gradients {
                    gradients[[n, l - 1]] = mp1p1p;
                }
                mp2p3p = mp1p1p;
                mp1p2p = mp00;
            }

            // makes sure l == 0 is taken care of
            if self.max_angular % 2 == 0 {
                let (a, b) = get_ab(n, 0);
                let m1p1p = g_gradient_recursive_step(a, b, z, m2p3p, m1p2p);
                let m00 = g_value_recursive_step(a, b, z, m1p2p, m1p1p);
                values[[n, 0]] = m00;
                if let Some(ref mut gradients) = gradients {
                    gradients[[n, 0]] = m1p1p;
                }
            }

            if let Some(ref mut gradients) = gradients {
                let mut row = gradients.index_axis_mut(Axis(0), n);
                row *= 2.0 * z / rij;
            }
        }

        if let Some(ref mut gradients) = gradients {
            azip!((gradient in gradients, &value in &values)
                *gradient -= 2.0 * alpha_rij * value
            );
        }
    }
}

/// single downward recursive step when compute G values
fn g_value_recursive_step(a: f64, b: f64, z: f64, m1p2p: f64, m1p1p: f64) -> f64 {
    return (z * (a - b) * m1p2p + m1p1p * b) / a;
}

/// single downward recursive step when compute G derivatives
fn g_gradient_recursive_step(_a: f64, b: f64, z: f64, m2p3p: f64, m1p2p: f64) -> f64 {
    return z * m2p3p + m1p2p * (b + 1.0);
}


#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hyp1f1_vs_gsl() {
        let all_z_values = [
            1e-3, 1e-2, 1e-1, 1.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 200.0, 500.0, 600.0
        ];

        for n in 0..20 {
            for l in 0..20 {
                let a = (n + l + 3) as f64 / 2.0;
                let b = l as f64 + 1.5;

                let hyper = HyperGeometric::new(a, b);
                for &z in &all_z_values {
                    let gsl_value = rgsl::hypergeometric::hyperg_1F1(a, b, z);
                    let hyper_value = hyper.compute_1f1(z);

                    assert_relative_eq!(
                        gsl_value, hyper_value,
                        epsilon=HYPERGEOMETRIC_PRECISION,
                        max_relative=1e1 * HYPERGEOMETRIC_PRECISION
                    );
                }
            }
        }
    }

    fn gsl_hypergeometric(
        max_radial: usize,
        max_angular: usize,
        parameters: HyperGeometricParameters,
        rij: f64
    ) -> (Array2<f64>, Array2<f64>) {
        use rgsl::hypergeometric::hyperg_1F1;
        use rgsl::gamma_beta::gamma::gamma;

        let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);
        let mut gradients = Array2::from_elem((max_radial, max_angular + 1), 0.0);
        for n in 0..max_radial {
            for l in 0..(max_angular + 1) {
                let a = (n + l + 3) as f64 / 2.0;
                let b = l as f64 + 1.5;
                let c = parameters.atomic_gaussian_constant;
                let d = parameters.gto_gaussian_constants[n];

                let z = c * c * rij * rij / (c + d);
                if z < 600.0 {
                    values[[n, l]] = gamma(a) / gamma(b) * f64::exp(-c * rij * rij) * hyperg_1F1(a, b, z);

                    let grad = 2.0 * a * c * c * rij / (b * (c + d)) * gamma(a) / gamma(b) * f64::exp(-c * rij * rij) * hyperg_1F1(a + 1.0, b + 1.0, z);
                    gradients[[n, l]] = grad - 2.0 * c * rij * values[[n, l]];
                } else {
                    // exp overflows while computing hyperg_1F1 in gsl, the
                    // corresponding value should be close to 0.0
                    values[[n, l]] = f64::NAN;
                    gradients[[n, l]] = f64::NAN;
                }
            }
        }

        return (values, gradients);
    }

    fn gto_gaussian_constants(max_radial: usize, cutoff: f64) -> Vec<f64> {
        let mut constants = Vec::new();
        for n in 0..max_radial {
            let sigma = cutoff * f64::sqrt(std::cmp::max(n, 1) as f64) / (max_radial as f64 + 1.0);
            constants.push(1.0 / (2.0 * sigma * sigma));
        }
        return constants;
    }

    #[test]
    fn hypergeometric_vs_gsl() {
        // this test goes to extreme values for most parameters, to check that we
        // can compute our hypergeometric function everywhere
        for &max_radial in &[0, 1, 2, 5, 6, 8, 9, 12, 15, 18, 19] {
            for &max_angular in &[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 15, 18, 19] {
                let hyper = HyperGeometricSphericalExpansion::new(max_radial, max_angular);
                let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);
                let mut gradients = Array2::from_elem((max_radial, max_angular + 1), 0.0);

                for atomic_gaussian_width in &[0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0, 10.0] {
                    for &cutoff in &[1.0, 2.0, 3.0, 5.0, 6.0, 20.0] {
                        let parameters = HyperGeometricParameters {
                            atomic_gaussian_constant: 1.0 / (2.0 * atomic_gaussian_width * atomic_gaussian_width),
                            gto_gaussian_constants: &gto_gaussian_constants(max_radial, cutoff),
                        };

                        for &rij in &[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 4.0, 5.0, 7.0, 10.0] {
                            if rij > cutoff {
                                continue;
                            }

                            hyper.compute(rij, parameters, values.view_mut(), Some(gradients.view_mut()));
                            let (gsl_values, gsl_gradients) = gsl_hypergeometric(max_radial, max_angular, parameters, rij);

                            for n in 0..max_radial {
                                for l in 0..(max_angular + 1) {
                                    if gsl_values[[n, l]].is_nan() {
                                        continue;
                                    }

                                    assert_relative_eq!(
                                        gsl_values[[n, l]], values[[n, l]],
                                        epsilon=HYPERGEOMETRIC_PRECISION,
                                        max_relative=1e2 * HYPERGEOMETRIC_PRECISION
                                    );

                                    assert_relative_eq!(
                                        gsl_gradients[[n, l]], gradients[[n, l]],
                                        epsilon=HYPERGEOMETRIC_PRECISION,
                                        // This is still 1e-7, but the precision
                                        // goes down a lot for gradients. I don't
                                        // know why.
                                        max_relative=1e6 * HYPERGEOMETRIC_PRECISION
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn finite_differences() {
        let delta = 1e-9;

        for &max_radial in &[0, 1, 2, 5, 6, 8, 9, 12] {
            for &max_angular in &[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12] {
                let hyper = HyperGeometricSphericalExpansion::new(max_radial, max_angular);
                let mut values = Array2::from_elem((max_radial, max_angular + 1), 0.0);
                let mut values_delta = Array2::from_elem((max_radial, max_angular + 1), 0.0);
                let mut gradients = Array2::from_elem((max_radial, max_angular + 1), 0.0);

                for atomic_gaussian_width in &[0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0] {
                    let parameters = HyperGeometricParameters {
                        atomic_gaussian_constant: 1.0 / (2.0 * atomic_gaussian_width * atomic_gaussian_width),
                        gto_gaussian_constants: &gto_gaussian_constants(max_radial, 3.5),
                    };

                    for &rij in &[0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 4.0, 5.0, 7.0, 10.0] {
                        hyper.compute(rij, parameters, values.view_mut(), Some(gradients.view_mut()));
                        hyper.compute(rij + delta, parameters, values_delta.view_mut(), None);

                        let finite_difference = (values_delta.clone() - values.clone()) / delta;

                        for n in 0..max_radial {
                            for l in 0..(max_angular + 1) {
                                assert_relative_eq!(
                                    gradients[[n, l]], finite_difference[[n, l]],
                                    epsilon=delta,
                                    max_relative=2e-3,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
