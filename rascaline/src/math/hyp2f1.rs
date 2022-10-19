#![allow(clippy::many_single_char_names, clippy::similar_names, clippy::too_many_lines)]
#![allow(clippy::float_cmp, clippy::if_same_then_else)]

use std::f64::consts::PI;

use crate::math::{digamma, gamma, EULER};

fn is_integer(x: f64) -> bool {
    return (x as i32) as f64 == x;
}

/// Compute the 2F1 hypergeometric function.
///
/// The current implementation does not support any of the 
/// following combination of input parameters
///
///  - `x == 1` and `c - a - b < 0`
///  - `c < 0` and `a < 0` and `a > c`
///  - `c < 0` and `b < 0` and `b > c` 
///
/// The implementation is translated from scipy's specfun.f, and distributed
/// under the CSD-3-Clauses license, Copyright (c) 2001-2002 Enthought, Inc.
/// 2003-2022, scipy Developers.
pub fn hyp2f1(a: f64, b: f64, c: f64, mut x: f64) -> f64 {
    assert!(a.is_finite() && b.is_finite() && c.is_finite() && x.is_finite());
    assert!(x < 1.0, "x must be smaller than 1.0");

    let c_is_neg_int = is_integer(c) && c < 0.0;
    let a_is_neg_int = is_integer(a) && a < 0.0;
    let b_is_neg_int = is_integer(b) && b < 0.0;
    let ca_is_neg_int = is_integer(c - a) && c - a <= 0.0;
    let cb_is_neg_int = is_integer(c - b) && c - b <= 0.0;

    if 1.0 - x < 1e-15 && c - a - b <= 0.0 && !(a_is_neg_int || b_is_neg_int) {
        // the condition in specfun (c_is_neg_int || 1.0 - x < 1e-15 && c - a -
        // b <= 0.0) is too strict here, so we are following the code from
        // cephes instead.
        panic!("invalid arguments to hyp2f1 (x==1 & c - a - b < 0)");
    } else if c_is_neg_int {
        if a_is_neg_int && a.round() > c.round() {
            // everything is fine
        } else if b_is_neg_int && b.round() > c.round() {
            // everything is fine
        } else {
            panic!("invalid arguments to hyp2f1 (c < 0)");
        }
    }

    let eps = if x > 0.95 {
        1e-8
    } else {
        1e-15
    };

    if x == 0.0 || a == 0.0 || b == 0.0 {
        return 1.0;
    } else if 1.0 - x == eps && c - a - b > 0.0 {
        let gamma_c = gamma(c);
        let gamma_cab = gamma(c - a - b);
        let gamma_ca = gamma(c - a);
        let gamma_cb = gamma(c - b);
        return gamma_c * gamma_cab / (gamma_ca * gamma_cb);
    } else if x + 1. <= eps && (c - a + b - 1.0).abs() <= eps {
        let gamma_0 = f64::sqrt(PI) * 2.0_f64.powf(-a);
        let gamma_1 = gamma(c);
        let gamma_2 = gamma(a / 2.0 + 1.0 - b);
        let gamma_3 = gamma(a * 0.5 + 0.5);
        return gamma_0 * gamma_1 / (gamma_2 * gamma_3);
    } else if a_is_neg_int || b_is_neg_int {
        let nm = if b_is_neg_int {
            b.abs()
        } else {
            a.abs()
        } as usize;

        let mut hf = 1.;
        let mut term = 1.;
        for k in (1..=nm).map(|k| k as f64) {
            term *= (a + k - 1.) * (b + k - 1.) / (k * (c + k - 1.)) * x;
            hf += term;
        }
        return hf;
    } else if ca_is_neg_int || cb_is_neg_int {
        let nm = if cb_is_neg_int {
            (c - b).abs()
        } else {
            (c - a).abs()
        } as usize;

        let mut hf = 1.;
        let mut term = 1.;
        for k in (1..=nm).map(|k| k as f64) {
            term *= (c - a + k - 1.) * (c - b + k - 1.) / (k * (c + k - 1.)) * x;
            hf += term;
        }

        return (1.0 - x).powf(c - a - b) * hf;
    }

    let aa = a;
    let bb = b;
    let x1 = x;

    let (a, b) = if x < 0.0 {
        x /= x - 1.;

        let (a_in, mut b_in) = if c > a && b < a && b > 0.0 {
            (bb, aa)
        } else {
            (aa, bb)
        };

        b_in = c - b_in;

        (a_in, b_in)
    } else {
        (a, b)
    };

    let mut result;
    if x >= 0.75 {
        if (c - a - b - ((c - a - b) as i64) as f64).abs() < 1e-15 {
            let mut m = (c - a - b) as i32;
            let gamma_a = gamma(a);
            let gamma_b = gamma(b);
            let gamma_c = gamma(c);
            let gamma_am = gamma(a + m as f64);
            let gamma_bm = gamma(b + m as f64);
            let digamma_a = digamma(a);
            let digamma_b = digamma(b);

            let mut gm = if m == 0 {0.0 } else { 1.0 };

            for j in (1..m.abs()).map(|j| j as f64) {
                gm *= j;
            }

            let mut rm = 1.0;
            for j in (1..=m.abs()).map(|j| j as f64) {
                rm *= j;
            }

            let mut f0 = 1.0;
            let mut r0 = 1.0;
            let mut r1 = 1.0;
            let mut sp0 = 0.0;
            let mut sp = 0.0;

            if m >= 0 {
                let c0 = gm * gamma_c / (gamma_am * gamma_bm);
                let c1 = -gamma_c * (x - 1.).powi(m) / (gamma_a * gamma_b * rm);
                for k in (1..m).map(|k| k as f64) {
                    r0 *= (a + k - 1.) * (b + k - 1.0) / (k * (k - m as f64)) * (1.0 - x);
                    f0 += r0;
                }
                for k in (1..=m).map(|k| k as f64) {
                    sp0 += 1. / (a + k - 1.0) + 1.0 / (b + k - 1.0) - 1.0 / k;
                }

                let mut f1 = digamma_a + digamma_b + sp0 + EULER * 2. + (1. - x).ln();

                let mut previous = 0.0;
                for k in (1..=250).map(|k| k as f64) {
                    sp += (1. - a) / (k * (a + k - 1.0)) + (1.0 - b) / (k * (b + k - 1.0));

                    let mut sm = 0.0;
                    for j in (1..=m).map(|k| k as f64) {
                        sm += (1. - a) / ((j + k) * (a + j + k - 1.0)) + 1.0 / (b + j + k - 1.0);
                    }

                    let rp = digamma_a + digamma_b + EULER * 2.0 + sp + sm + (1.0 - x).ln();
                    r1 *= (a + m as f64 + k - 1.0) * (b + m as f64 + k - 1.0) / (k * (m as f64 + k)) * (1.0 - x);
                    f1 += r1 * rp;
                    if (f1 - previous).abs() < (f1).abs() * eps {
                        break;
                    }
                    previous = f1;
                }

                result = f0 * c0 + f1 * c1;

            } else {
                m *= -1;
                let c0 = gm * gamma_c / (gamma_a * gamma_b * (1. - x).powi(m));
                let c1 = -((-1.0_f64).powi(m)) * gamma_c / (gamma_am * gamma_bm * rm);

                for k in (1..m).map(|k| k as f64) {
                    r0 *= (a - m as f64 + k - 1.) * (b - m as f64 + k - 1.0) / (k * (k - m as f64)) * (1.0 - x);
                    f0 += r0;
                }

                for k in (1..=m).map(|k| k as f64) {
                    sp0 += 1. / k;
                }

                let mut f1 = digamma_a + digamma_b - sp0 + EULER * 2. + (1.0 - x).ln();

                let mut previous = 0.0;
                for k in (1..=250).map(|k| k as f64) {
                    sp += (1. - a) / (k * (a + k - 1.0)) + (1.0 - b) / (k * (b + k - 1.0));
                    let mut sm = 0.;
                    for j in (1..m+1).map(|j| j as f64) {
                        sm += 1. / (j + k);
                    }
                    let rp = digamma_a + digamma_b + EULER * 2. + sp - sm + (1. - x).ln();
                    r1 *= (a + k - 1.0) * (b + k - 1.0) / (k * (m as f64 + k)) * (1.0 - x);
                    f1 += r1 * rp;
                    if (f1 - previous).abs() < (f1).abs() * eps {
                        break;
                    }
                    previous = f1;
                }

                result = f0 * c0 + f1 * c1;
            }
        } else {
            let gamma_a = gamma(a);
            let gamma_b = gamma(b);
            let gamma_c = gamma(c);
            let gamma_ca = gamma(c - a);
            let gamma_cb = gamma(c - b);
            let gamma_cab = gamma(c - a - b);
            let gamma_abc = gamma(a + b - c);
            let c0 = gamma_c * gamma_cab / (gamma_ca * gamma_cb);
            let c1 = gamma_c * gamma_abc / (gamma_a * gamma_b) * (1.0 - x).powf(c - a - b);

            result = 0.;
            let mut r0 = c0;
            let mut r1 = c1;
            let mut previous = 0.0;
            for k in (1..=250).map(|k| k as f64) {
                r0 *= (a + k - 1.) * (b + k - 1.0) / (k * (a + b - c + k)) * (1.0 - x);
                r1 *= (c - a + k - 1.) * (c - b + k - 1.0) / (k * (c - a - b + k)) * (1.0 - x);
                result += r0 + r1;
                if (result - previous).abs() < (result).abs() * eps {
                    break;
                }
                previous = result;
            }
            result += c0 + c1;
        };
    } else {
        let (a0, a, b, ) = if c > a && c < a * 2.0 && c > b && c < b * 2.0 {
            let a0 = (1.0 - x).powf(c - a - b);
            (a0, c - a, c - b)
        } else {
            (1.0, a, b)
        };

        result = 1.;
        let mut previous = 0.0;
        let mut term = 1.;
        for k in (1..=250).map(|k| k as f64) {
            term *= (a + k - 1.) * (b + k - 1.) / (k * (c + k - 1.)) * x;
            result += term;
            if (result - previous).abs() <= (result).abs() * eps {
                break;
            }
            previous = result;
        }
        result *= a0;
    }

    if x1 < 0.0 {
        let c0 = 1.0 / (1. - x1).powf(aa);
        result *= c0;
    }

    return result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // all the reference values (a, b, c, x, expected) were computed using mpmath
    #[test]
    fn hyp2f1_scipy_points() {
        // some special points taken from scipy test suite
        let scipy_points = [
            (1.0, 2.0, 3.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, -1.0, 1.0),
            (0.1, 1.0, 2.0, -5.0, 0.892389513881694),
            // (-2.0, -2.0, -3.0, 0.5, 0.416666666666667),  // invalid arguments
            (-7.2, -0.5, -4.5, 1e-05, 0.999991999964571),
            // (1.0, 2.0, -3.0, -1.0, std::f64::INFINITY),  // invalid arguments
            (1.0, 0.5, -4.5, -20.0, 0.446746825239319),
            (-1.0, -2.0, 3.0, 0.0, 1.0),
            (2.0, -2.0, -3.0, 0.5, 1.91666666666667),
            // (2.0, -3.0, -2.0, -1.0, std::f64::INFINITY), // invalid arguments
            (-7.2, 0.5, -4.5, 1e-05, 1.00000800010629),
            (-1.0, 2.0, 3.0, 0.0, 1.0),
            (-1.0, -2.0, 4.0, -1.0, 0.5),
            // (0.0, -6.0, -4.0, -2.0, 1.0),                // invalid arguments
            (0.0, -3.0, -4.0, -1.0, 1.0),
            // (1.0, -6.0, -4.0, -1.0, std::f64::INFINITY), // invalid arguments
            (1.0, -0.5, -4.5, -5.0, 0.469756352472402),
            (-7.2, -0.5, -4.5, -1.0, 19.9158826472958),
            (-1.0, -1.0, -2.0, -1.0, 1.5),
            (-1.0, -1.0, -2.0, 0.0, 1.0),
            (-1.0, -1.0, -2.0, 0.9, 0.55),
            // (-1.0, -1.0, -1.0, -1.0, 2.0),               // invalid arguments
            // (-1.0, -1.0, -1.0, 0.0, 1.0),                // invalid arguments
            // (-1.0, -1.0, -1.0, 0.9, 0.1),                // invalid arguments
            // (-1.0, -1.0, 0.0, 0.0, std::f64::NAN),       // gives 1.0
            (-1.0, 0.0, -2.0, -1.0, 1.0),
            (-1.0, 0.0, -2.0, 0.0, 1.0),
            (-1.0, 0.0, -2.0, 0.9, 1.0),
            // (-1.0, 0.0, -1.0, -1.0, 1.0),                // invalid arguments
            // (-1.0, 0.0, -1.0, 0.0, 1.0),                 // invalid arguments
            // (-1.0, 0.0, -1.0, 0.9, 1.0),                 // invalid arguments
            (-1.0, 0.0, 0.0, -1.0, 1.0),
            (-1.0, 0.0, 0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0, 0.9, 1.0),
            (-1.0, 1.0, -2.0, -1.0, 0.5),
            (-1.0, 1.0, -2.0, 0.0, 1.0),
            (-1.0, 1.0, -2.0, 0.9, 1.45),
            // (-1.0, 1.0, -1.0, 0.0, 1.0),                 // invalid arguments
            // (-1.0, 1.0, -1.0, 0.9, 1.9),                 // invalid arguments
            // (-1.0, 1.0, 0.0, 0.0, std::f64::NAN),        // gives 1.0
            (0.0, -1.0, -2.0, -1.0, 1.0),
            (0.0, -1.0, -2.0, 0.0, 1.0),
            (0.0, -1.0, -2.0, 0.9, 1.0),
            // (0.0, -1.0, -1.0, -1.0, 1.0),                // invalid arguments
            // (0.0, -1.0, -1.0, 0.0, 1.0),                 // invalid arguments
            // (0.0, -1.0, -1.0, 0.9, 1.0),                 // invalid arguments
            (0.0, -1.0, 0.0, -1.0, 1.0),
            (0.0, -1.0, 0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0, 0.9, 1.0),
            // (0.0, 0.0, -2.0, -1.0, 1.0),                 // invalid arguments
            // (0.0, 0.0, -2.0, 0.0, 1.0),                  // invalid arguments
            // (0.0, 0.0, -2.0, 0.9, 1.0),                  // invalid arguments
            // (0.0, 0.0, -1.0, -1.0, 1.0),                 // invalid arguments
            // (0.0, 0.0, -1.0, 0.0, 1.0),                  // invalid arguments
            // (0.0, 0.0, -1.0, 0.9, 1.0),                  // invalid arguments
            (0.0, 0.0, 0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.9, 1.0),
            // (0.0, 1.0, -2.0, -1.0, 1.0),                 // invalid arguments
            // (0.0, 1.0, -2.0, 0.0, 1.0),                  // invalid arguments
            // (0.0, 1.0, -2.0, 0.9, 1.0),                  // invalid arguments
            // (0.0, 1.0, -1.0, -1.0, 1.0),                 // invalid arguments
            // (0.0, 1.0, -1.0, 0.0, 1.0),                  // invalid arguments
            // (0.0, 1.0, -1.0, 0.9, 1.0),                  // invalid arguments
            (0.0, 1.0, 0.0, -1.0, 1.0),
            (0.0, 1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 0.9, 1.0),
            (1.0, -1.0, -2.0, -1.0, 0.5),
            (1.0, -1.0, -2.0, 0.0, 1.0),
            (1.0, -1.0, -2.0, 0.9, 1.45),
            // (1.0, -1.0, -1.0, 0.0, 1.0),                 // invalid arguments
            // (1.0, -1.0, -1.0, 0.9, 1.9),                 // invalid arguments
            // (1.0, -1.0, 0.0, 0.0, std::f64::NAN),        // gives 1.0
            // (1.0, 0.0, -2.0, -1.0, 1.0),                 // invalid arguments
            // (1.0, 0.0, -2.0, 0.0, 1.0),                  // invalid arguments
            // (1.0, 0.0, -2.0, 0.9, 1.0),                  // invalid arguments
            // (1.0, 0.0, -1.0, -1.0, 1.0),                 // invalid arguments
            // (1.0, 0.0, -1.0, 0.0, 1.0),                  // invalid arguments
            // (1.0, 0.0, -1.0, 0.9, 1.0),                  // invalid arguments
            (1.0, 0.0, 0.0, -1.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.9, 1.0),
            // (1.0, 1.0, -2.0, -1.0, std::f64::INFINITY),  // invalid arguments
            // (1.0, 1.0, -2.0, 0.0, 1.0),                  // invalid arguments
            // (1.0, 1.0, -2.0, 0.9, std::f64::INFINITY),   // invalid arguments
            // (1.0, 1.0, -1.0, -1.0, std::f64::INFINITY),  // invalid arguments
            // (1.0, 1.0, -1.0, 0.0, 1.0),                  // invalid arguments
            // (1.0, 1.0, -1.0, 0.9, std::f64::INFINITY),   // invalid arguments
            // (1.0, 1.0, 0.0, 0.0, std::f64::NAN),         // gives 1.0
        ];

        for (a, b, c, x, expected) in scipy_points {
            dbg!(a, b, c, x);
            assert_relative_eq!(hyp2f1(a, b, c, x), expected, max_relative=1e-12, epsilon=1e-15);
        }
    }

    #[test]
    fn hyp2f1_lode_points() {
        // some points from our expected usage of hyp2f1 (center
        // atom contribution in LODE). This uses `a=p/2`, `b=(n + 3)/2`,
        // `c=(p+2)/2`, and `x` from -1e-3 to -1e3.
        let actual_values = [
            (0.5, 1.5, 1.5, 0.23, 1.13960576459638),
            (0.5, 1.5, 1.5, 0.99, 10.0),
            (0.5, 1.5, 1.5, -0.001, 0.999500374687773),
            (0.5, 1.5, 1.5, -0.03, 0.985329278164293),
            (0.5, 1.5, 1.5, -42.0, 0.152498570332605),
            (0.5, 1.5, 1.5, -1000.0, 0.0316069770620507),
            (0.5, 4.5, 1.5, -0.001, 0.998502471175221),
            (0.5, 4.5, 1.5, -0.03, 0.957128423470113),
            (0.5, 4.5, 1.5, -42.0, 0.0705386685734078),
            (0.5, 4.5, 1.5, -1000.0, 0.0144561264464801),
            (1.5, 1.5, 2.5, -0.001, 0.999100802842932),
            (1.5, 1.5, 2.5, -0.03, 0.973704055664767),
            (1.5, 1.5, 2.5, -42.0, 0.017409586550297),
            (1.5, 1.5, 2.5, -1000.0, 0.000298624095345665),
            (1.5, 4.5, 2.5, -0.001, 0.99730529464762),
            (1.5, 4.5, 2.5, -0.03, 0.92354254735985),
            (1.5, 4.5, 2.5, -42.0, 0.000839723220491458),
            (1.5, 4.5, 2.5, -1000.0, 7.22806320748387e-6),
            (3.0, 1.5, 4.0, -0.001, 0.998876123907304),
            (3.0, 1.5, 4.0, -0.03, 0.967233799071644),
            // (3.0, 1.5, 4.0, -42.0, 0.00675325908887205),
            // (3.0, 1.5, 4.0, -1000.0, 6.2976592595972e-5),
            (3.0, 4.5, 4.0, 0.23, 2.42269724200498),
            (3.0, 4.5, 4.0, 0.99, 8588515.61661476),
            (3.0, 4.5, 4.0, -0.001, 0.996632411615264),
            (3.0, 4.5, 4.0, -0.03, 0.905087236117485),
            // (3.0, 4.5, 4.0, -42.0, 6.07718118072915e-6),
            // (3.0, 4.5, 4.0, -1000.0, 4.57079782017785e-10),
        ];

        for (a, b, c, x, expected) in actual_values {
            let max_relative = if a >= 3.0 {
                5e-2
            } else {
                1e-6
            };
            assert_relative_eq!(hyp2f1(a, b, c, x), expected, max_relative=max_relative, epsilon=1e-15);
        }
    }
}
