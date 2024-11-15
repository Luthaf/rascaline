#![allow(clippy::redundant_else)]

//! Provides the exponential integral functions
//!
//! This code was converted from scipy specfun.f

/// Compute exponential integral E1(x)
pub fn exp1(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    } else if x <= 1.0 {
        let mut e1 = 1.0;
        let mut r = 1.0;
        for k in 1..=25 {
            let d1 = k as f64 + 1.;
            r = -r * k as f64 * x / (d1 * d1);
            e1 += r;
            if f64::abs(r) <= f64::abs(e1) * 1e-15 {
                break;
            }
        }

        return -0.5772156649015328 - f64::ln(x) + x * e1;
    } else {
        let m = ((80.0 / x) as usize) + 20;
        let mut t0 = 0.;
        for k in (1..=m).rev() {
            t0 = k as f64 / (k as f64 / (x + t0) + 1.);
        }
        let t = 1.0 / (x + t0);
        return f64::exp(-(x)) * t;
    }
}

/// Compute exponential integral Ei(x)
pub fn expi(x: f64) -> f64 {
    if x == 0.0 {
        return -f64::INFINITY;
    } else if x < 0.0 {
        return -exp1(-x);
    } else if f64::abs(x) <= 40.0 {
        // Power series around x = 0
        let mut ei = 1.;
        let mut r = 1.;
        for k in 1..=100 {
            let d = k as f64 + 1.;
            r = r * k as f64 * x / (d * d);
            ei += r;
            if f64::abs(r / ei) <= 1e-15 {
                break;
            }
        }

        return 0.5772156649015328 + f64::ln(x) + x * ei;
    } else {
        // Asymptotic expansion (the series is not convergent)
        let mut ei = 1.;
        let mut r = 1.;
        for k in 1..=20 {
            r = r * k as f64 / x;
            ei += r;
        }
        return f64::exp(x) / x * ei;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exp1() {
        // Reference values from scipy.special.exp1
        assert!(exp1(-1.).is_nan());
        assert_eq!(exp1(0.), f64::INFINITY);
        assert_relative_eq!(exp1(1e-05), 10.935719800043696, max_relative=1e-15);
        assert_relative_eq!(exp1(0.33), 0.8361011614550026, max_relative=1e-15);
        assert_relative_eq!(exp1(1.), 0.21938393439552062, max_relative=1e-15);
        assert_relative_eq!(exp1(2.5), 0.024914917870269736, max_relative=1e-15);
        assert_relative_eq!(exp1(43.), 4.809496556950017e-21, max_relative=1e-15);
    }

    #[test]
    fn test_expi() {
        // Reference values from scipy.special.expi
        assert_relative_eq!(expi(-42.), -1.3377908810011776e-20, max_relative=1e-15);
        assert_relative_eq!(expi(-1.), -0.21938393439552062, max_relative=1e-15);
        assert_eq!(expi(0.), -f64::INFINITY);
        assert_relative_eq!(expi(1e-05), -10.935699800043697, max_relative=1e-15);
        assert_relative_eq!(expi(0.33), -0.1720950921354428, max_relative=1e-15);
        assert_relative_eq!(expi(1.), 1.8951178163559368, max_relative=1e-15);
        assert_relative_eq!(expi(2.5), 7.073765894578603, max_relative=1e-15);
        assert_relative_eq!(expi(43.), 1.1263482901669605e17, max_relative=1e-15);
    }
}
