use std::f64;

use crate::math::ln_gamma;

use super::gamma;

// Accepted relative error for hyp1f1
const ACCEPTABLE_RTOL: f64 = 1e-9;


/// Compute the 1F1 confluent hypergeometric function.
///
/// The implementation is translated from scipy, and distributed under the
/// CSD-3-Clauses license, Copyright (c) 2001-2002 Enthought, Inc. 2003-2022,
/// scipy Developers.
#[allow(clippy::float_cmp)]
pub fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    assert!(a.is_finite() && b.is_finite() && x.is_finite());
    assert!(b > 0.0, "b must be positive");

    if a == 0.0 || x == 0.0 {
        return 1.0;
    } else if a == -1.0 {
        return 1.0 - x / b;
    } else if a == b {
        return f64::exp(x);
    } else if a - b == 1.0 {
        return (1.0 + x / b) * f64::exp(x);
    } else if a <= 0.0 && a == f64::floor(a) {
        // The geometric series is finite in this case, but it could
        // still suffer from cancellation.
        return hyp1f1_series_track_convergence(a, b, x)
    }

    if b > 0.0 && (f64::abs(a) + 1.0) * f64::abs(x) < 0.9 * b {
        // For the kth term of the series we are multiplying by
        //
        // t_k = (a + k) * x / ((b + k) * (k + 1))
        //
        // We have that
        //
        // |t_k| < (|a| + 1) * |x| / |b|,
        //
        // which means that in this branch we get geometric
        // convergence.
        return hyp1f1_series(a, b, x);
    }

    return chgm_fortran(a, b, x);
}


fn hyp1f1_series_track_convergence(a: f64, b: f64, x: f64) -> f64 {
    // The hypergeometric series can suffer from cancellation or take a
    // prohibitive number of terms to converge. This function computes
    // the series while monitoring those conditions.
    let mut converged = false;
    let mut n_steps_to_converge = 0.0;

    let mut a_p_k;
    let mut b_p_k;
    let mut term = 1.0;
    let mut result = 1.0;
    let mut abssum = result;

    for k in (0..1000).map(|k| k as f64) {
        a_p_k = a + k;
        b_p_k = b + k;
        if b_p_k != 0.0 {
            term *= a_p_k * x / b_p_k / (k + 1.0);
        } else if a_p_k == 0.0 {
            // The Pochammer symbol in the denominator has become zero,
            // but we still have the continuation formula DLMF 13.2.5.
            term = 0.0;
        } else {
            // We hit a pole
            return f64::NAN;
        }

        abssum += f64::abs(term);
        result += term;
        if f64::abs(term) <= f64::EPSILON * f64::abs(result) {
            n_steps_to_converge = k;
            converged = true;
            break;
        }
    }

    if !converged {
        return f64::NAN;
    }

    if n_steps_to_converge * f64::EPSILON * abssum <= ACCEPTABLE_RTOL * f64::abs(result) {
        return result;
    }

    return f64::NAN;
}


fn hyp1f1_series(a: f64, b: f64, x: f64) -> f64 {
    let mut term = 1.0;
    let mut result = 1.0;

    let mut converged = false;
    for k in (0..1000).map(|k| k as f64) {
        term *= (a + k) * x / (b + k) / (k + 1.0);
        result += term;
        if f64::abs(term) <= f64::EPSILON * f64::abs(result) {
            converged = true;
            break;
        }
    }

    if !converged {
        return f64::NAN;
    }

    return result
}

/// This is a rust version of the CHGM fortran subroutine in scipy
///
/// The original fortran source was translated to C with f2c, then manually
/// cleaned up and translated to rust.
///
/// DLMF refers to <https://dlmf.nist.gov/>
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn chgm_fortran(mut a: f64, b: f64, mut x: f64) -> f64{
    let mut a0;
    let mut y0;
    let mut y1;
    let mut la;
    let mut nl;

    let mut hg1;
    let mut hg2;
    let mut ln_gamma_a;
    let mut ln_gamma_b;
    let mut ln_gamma_ba;

    a0 = a;
    let x0 = x;

    let mut result = 0.0;

    /* DLMF 13.2.39 */
    if x < 0. {
        a = b - a;
        a0 = a;
        x = f64::abs(x);
    }
    nl = 0;
    la = 0;
    if a >= 2.0 {
        /* preparing terms for DLMF 13.3.1 */
        nl = 1;
        la = a as i32;
        a = a - (la as f64) - 1.0;
    }

    y0 = 0.0;
    y1 = 0.0;

    for n in 0..=nl {
        if a0 >= 2.0 {
            a += 1.0;
        }
        if x <= f64::abs(b) + 30.0 || a < 0.0 {
            result = 1.0;
            let mut term = 1.0;

            if x < 500.0 {
                // use the normal definition (DLMF 13.2.2)
                for j in (1..=500).map(|j| j as f64) {
                    term *= (a + j - 1.) / (j * (b + j - 1.)) * x;
                    result += term;
                    if result != 0. && f64::abs(term / result) < 1e-15 {
                        break;
                    }
                }

                /* DLMF 13.2.39 (cf. above) */
                if x0 < 0.0 {
                    result *= f64::exp(x0);
                }
            } else {
                // use the definition for x -> \infty (DLMF 13.7.1 and 13.2.3)
                for j in (1..=50).map(|j| j as f64) {
                    term *= (j - a) * (b - a + j - 1.) / (j * x);
                    result += term;
                    if result != 0. && f64::abs(term / result) < 1e-15 {
                        break;
                    }
                }

                result *= x.powf(a - b) * gamma(b) / gamma(a);
                /* DLMF 13.2.39 cancels the exp(x) for x0 < 0 */
                if x0 > 0.0 {
                    result *= f64::exp(x);
                }
            }
        } else {
            /* DLMF 13.7.2 & 13.2.4, SUM2 corresponds to first sum */
            ln_gamma_a = ln_gamma(a);
            ln_gamma_b = ln_gamma(b);
            ln_gamma_ba = ln_gamma(b - a);

            let mut sum_1 = 1.0;
            let mut sum_2 = 1.0;
            let mut term_1 = 1.0;
            let mut term_2 = 1.0;

            // CHANGED from the original FORTRAN implementation
            //
            // use 30 terms instead of 8 here, to get better accuracy (accuracy
            // on GTO goes from ~1e-6 for gradients to ~1e-11)
            for i in (1..=30).map(|i| i as f64) {
                term_1 = -term_1 * (a + i - 1.) * (a - b + i) / (x * i);
                term_2 = -term_2 * (b - a + i - 1.) * (a - i) / (x * i);
                sum_1 += term_1;
                sum_2 += term_2;
            }

            if x0 >= 0.0 {
                hg1 = f64::exp(ln_gamma_b - ln_gamma_ba) * f64::powf(x, -a) * f64::cos(f64::consts::PI * a) * sum_1;
                hg2 = f64::exp(ln_gamma_b - ln_gamma_a + x) * f64::powf(x, a - b) * sum_2;
            } else {
                /* DLMF 13.2.39 (cf. above) */
                hg1 = f64::exp(ln_gamma_b - ln_gamma_ba + x0) * f64::powf(x, -a) * f64::cos(f64::consts::PI * a) * sum_1;
                hg2 = f64::exp(ln_gamma_b - ln_gamma_a) * f64::powf(x, a - b) * sum_2;
            }
            result = hg1 + hg2;
        }

        if n == 0 {
            y0 = result;
        }
        if n == 1 {
            y1 = result;
        }
    }

    if a0 >= 2.0 {
        /* DLMF 13.3.1 */
        for _ in 1..la {
            result = ((a * 2.0 - b + x) * y1 + (b - a) * y0) / a;
            y0 = y1;
            y1 = result;
            a += 1.0;
        }
    }

    return result;
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    static ALL_Z: [f64; 13] = [
        0.0, 1.3, 42.0, 200.0, 660.0, -1.0, -40.0, -200.0, -500.0, 3e-1, 2e-4, -4e-1, -8e-4,
    ];

    #[allow(clippy::excessive_precision)]
    // This was computed using mpmath
    static REFERENCE: [((u32, u32), [f64; 13]); 6] = [
        ((0, 0), [1.0, 3.66929666761924, 1.7392749415205e+18, 7.22597376812575e+86, 4.30881706558659e+286, 0.367879441171442, 4.24835425529159e-18, 1.38389652673674e-87, 7.12457640674129e-218, 1.349858807576, 1.00020002000133, 0.670320046035639, 0.999200319914684]),
        ((1, 1), [1.0, 3.66929666761924, 1.7392749415205e+18, 7.22597376812575e+86, 4.30881706558659e+286, 0.367879441171442, 4.24835425529159e-18, 1.38389652673674e-87, 7.12457640674129e-218, 1.349858807576, 1.00020002000133, 0.670320046035639, 0.999200319914684]),
        ((5, 6), [1.0, 3.38401720456627, 6.51794558745076e+17, 1.3083980191885e+86, 4.33929373469791e+285, 0.394775712369039, 7.11266970817605e-9, 8.39682564330044e-14, 1.36092469650155e-16, 1.32355791582708, 1.0001866842364, 0.688849034868493, 0.999253614360372]),
        ((13, 12), [1.0, 3.8421649506219, 3.54519102852153e+18, 2.89704505824636e+87, 3.07058123688245e+287, 0.354011483088197, -3.5964781936063e-14, -3.28638636385084e-24, -8.25083453006346e-30, 1.36478040300447, 1.00020742886482, 0.660319994010777, 0.999170713570911]),
        ((12, 15), [1.0, 3.27284968979622, 2.44964775706581e+17, 1.37335603673345e+85, 1.46589177252824e+284, 0.403871683240152, 4.2979422347514e-12, 1.71877602004918e-22, 1.88996873825758e-28, 1.31381868447175, 1.00018183480621, 0.695409738903967, 0.99927299318159]),
        ((20, 20), [1.0, 3.66929666761924, 1.7392749415205e+18, 7.22597376812575e+86, 4.30881706558659e+286, 0.367879441171442, 4.24835425529159e-18, 1.38389652673674e-87, 7.12457640674129e-218, 1.349858807576, 1.00020002000133, 0.670320046035639, 0.999200319914684]),
    ];

    #[test]
    fn check_values() {
        for ((n, l), values) in REFERENCE {
            for (&z, expected) in ALL_Z.iter().zip(values) {
                let a = 0.5 * (n + l + 3) as f64;
                let b = l as f64 + 1.5;

                assert_relative_eq!(hyp1f1(a, b, z), expected, max_relative=1e-13);
            }
        }
    }

    #[test]
    fn finite_differences() {
        let delta = 1e-6;

        for &n in &[1, 5, 10, 18] {
            for &l in &[0, 2, 8, 15] {
                let a = 0.5 * (n + l + 3) as f64;
                let b = l as f64 + 1.5;

                for &z in &[-200.0, -10.0, -1.1, -1e-2, 0.2, 1.5, 10.0, 40.0, 523.0] {
                    let value = hyp1f1(a, b, z);
                    let gradient = a / b * hyp1f1(a + 1.0, b + 1.0, z);
                    let value_delta = hyp1f1(a, b, z + delta);

                    let finite_difference = (value_delta - value) / delta;

                    assert_relative_eq!(
                        gradient, finite_difference, epsilon=delta, max_relative=1e-5,
                    );
                }
            }
        }
    }
}
