use crate::math::consts::SQRT_FRAC_2_PI;
use std::f64::consts::PI;

use super::{eval_polynomial, eval_polynomial_1};

const Z1: f64 = 1.4681970642123893e1;
const Z2: f64 = 4.92184563216946e1;

/// Bessel function of the first kind, order one
///
/// The domain is divided into the intervals [0, 8] and (8, infinity). In the
/// first interval a 24 term Chebyshev expansion is used. In the second, the
/// asymptotic trigonometric representation is employed using two rational
/// functions of degree 5/5.
///
/// # Accuracy
/// ```text
///                      Absolute error:
/// arithmetic   domain      # trials      peak         rms
///    IEEE      0, 30       30000       2.6e-16     1.1e-16
/// ```
#[allow(clippy::many_single_char_names)]
pub fn bessel_j1(x: f64) -> f64 {
    let z;
    let mut p;

    let mut w = x;
    if x < 0.0 {
        return -bessel_j1(-x);
    }
    if w <= 5.0 {
        z = x * x;
        w = eval_polynomial(z, &RP) / eval_polynomial_1(z, &RQ);
        w = w * x * (z - Z1) * (z - Z2);
        return w;
    }
    w = 5.0 / x;
    z = w * w;
    p = eval_polynomial(z, &PP) / eval_polynomial(z, &PQ);
    let q = eval_polynomial(z, &QP) / eval_polynomial_1(z, &QQ);
    let xn = x - 0.75 * PI;
    p = p * f64::cos(xn) - w * q * f64::sin(xn);
    return p * SQRT_FRAC_2_PI / f64::sqrt(x);
}

/// Bessel function of second kind, order one
///
/// Returns Bessel function of the second kind of order one of the argument.
///
/// The domain is divided into the intervals [0, 8] and (8, infinity). In the
/// first interval a 25 term Chebyshev expansion is used, and a call to j1() is
/// required. In the second, the asymptotic trigonometric representation is
/// employed using two rational functions of degree 5/5.
///
/// # Accuracy
///
/// ```text
///                      Absolute error:
/// arithmetic   domain      # trials      peak         rms
///    IEEE      0, 30       30000       1.0e-15     1.3e-16
/// ```
///
/// (error criterion relative when |y1| > 1).
#[allow(clippy::many_single_char_names)]
pub fn bessel_y1(x: f64) -> f64 {
    let mut w;
    let z;
    let mut p;
    if x <= 5.0 {
        if x == 0.0 {
            // sf_error(SF_ERROR_SINGULAR);
            return -std::f64::INFINITY;
        } else if x <= 0.0 {
            // sf_error(SF_ERROR_DOMAIN);
            return std::f64::NAN;
        }
        z = x * x;
        w = x * (eval_polynomial(z, &YP) / eval_polynomial_1(z, &YQ));
        w += 2.0 / PI * (bessel_j1(x) * f64::ln(x) - 1.0 / x);
        return w;
    }
    w = 5.0 / x;
    z = w * w;
    p = eval_polynomial(z, &PP) / eval_polynomial(z, &PQ);
    let q = eval_polynomial(z, &QP) / eval_polynomial_1(z, &QQ);
    let xn = x - 0.75 * PI;
    p = p * f64::sin(xn) + w * q * f64::cos(xn);
    return p * SQRT_FRAC_2_PI / f64::sqrt(x);
}

static RP: [f64; 4] = [
    -8.999712257055594e8,
    4.5222829799819403e11,
    -7.274942452218183e13,
    3.682957328638529e15,
];

static RQ: [f64; 8] = [
    6.208364781180543e2,
    2.5698725675774884e5,
    8.351467914319493e7,
    2.215115954797925e10,
    4.749141220799914e12,
    7.843696078762359e14,
    8.952223361846274e16,
    5.322786203326801e18,
];

static PP: [f64; 7] = [
    7.621256162081731e-4,
    7.313970569409176e-2,
    1.1271960812968493,
    5.112079511468076,
    8.424045901417724,
    5.214515986823615,
    1.0,
];

static PQ: [f64; 7] = [
    5.713231280725487e-4,
    6.884559087544954e-2,
    1.105142326340617,
    5.073863861286015,
    8.399855543276042,
    5.209828486823619,
    1.0,
];

static QP: [f64; 8] = [
    5.108625947501766e-2,
    4.982138729512334,
    7.582382841325453e1,
    3.667796093601508e2,
    7.108563049989261e2,
    5.974896124006136e2,
    2.1168875710057213e2,
    2.5207020585802372e1,
];

static QQ: [f64; 7] = [
    7.423732770356752e1,
    1.0564488603826283e3,
    4.986410583376536e3,
    9.562318924047562e3,
    7.997041604473507e3,
    2.8261927851763908e3,
    3.360936078106983e2,
];

static YP: [f64; 6] = [
    1.2632047479017804e9,
    -6.473558763791603e11,
    1.1450951154182373e14,
    -8.127702555013251e15,
    2.024394757135949e17,
    -7.788771962659501e17,
];

static YQ: [f64; 8] = [
    5.943015923461282e2,
    2.3556409294306856e5,
    7.348119444597217e7,
    1.8760131610870617e10,
    3.8823127749623857e12,
    6.205577271469538e14,
    6.871410873553005e16,
    3.9727060811656064e18,
];

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_j1() {
        // reference computed with mpmath
        assert_relative_eq!(bessel_j1(-123.985), 0.04508621393470377, max_relative=1e-12);
        assert_relative_eq!(bessel_j1(-5.1), 0.3370972020182318);
        assert_relative_eq!(bessel_j1(-3.0), -0.3390589585259365);
        assert_relative_eq!(bessel_j1(0.0), 0.0);
        assert_relative_eq!(bessel_j1(0.00245), 0.0012249990808674174);
        assert_relative_eq!(bessel_j1(2.1752), 0.5593771605955342, max_relative=1e-12);
        assert_relative_eq!(bessel_j1(2345.13), 0.010822488420270095, max_relative=1e-12);
    }

    #[test]
    fn test_y1() {
        // reference computed with mpmath
        assert!(bessel_y1(-123.985).is_nan());
        assert!(bessel_y1(-5.1).is_nan());
        assert!(bessel_y1(-3.0).is_nan());
        assert_relative_eq!(bessel_y1(0.0), -f64::INFINITY);
        assert_relative_eq!(bessel_y1(0.00245), -259.84997363769, max_relative=1e-12);
        assert_relative_eq!(bessel_y1(2.1752), -0.0114834540278188, max_relative=1e-12);
        assert_relative_eq!(bessel_y1(2345.13), -0.0124232991092643, max_relative=1e-12);
    }
}
