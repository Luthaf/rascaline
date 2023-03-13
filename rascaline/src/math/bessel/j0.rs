use super::{eval_polynomial, eval_polynomial_1};

/* 5.783185962946784521175995758455807035071 */
const DR1: f64 = 5.783185962946784;
/* 30.47126234366208639907816317502275584842 */
const DR2: f64 = 30.471262343662087;

use std::f64::consts::{FRAC_PI_4, PI};
use crate::math::consts::SQRT_FRAC_2_PI;

/// Bessel function of the first kind, order zero.
///
/// The domain is divided into the intervals [0, 5] and
/// (5, infinity). In the first interval the following rational
/// approximation is used:
///
/// ```text
///        2         2
/// (w - r  ) (w - r  ) P (w) / Q (w)
///       1         2    3       8
/// ```
///
/// where w = x^2  and the two r's are zeros of the function.
///
/// In the second interval, the Hankel asymptotic expansion
/// is employed with two rational functions of degree 6/6
/// and 7/7.
///
/// # Accuracy
/// ```text
///                      Absolute error:
/// arithmetic   domain     # trials      peak         rms
///    IEEE      0, 30       60000       4.2e-16     1.1e-16
/// ```
pub fn bessel_j0(mut x: f64) -> f64 {
    if x < 0.0 {
        x = -x;
    }

    if x <= 5.0 {
        let z = x * x;
        if x < 1e-5 {
            return 1.0 - z / 4.0;
        }
        let p = (z - DR1) * (z - DR2);
        return p * eval_polynomial(z, &RP) / eval_polynomial_1(z, &RQ);
    }

    let w = 5.0 / x;
    let q = 25.0 / (x * x);
    let p = eval_polynomial(q, &PP) / eval_polynomial(q, &PQ);
    let q = eval_polynomial(q, &QP) / eval_polynomial_1(q, &QQ);
    let xn = x - FRAC_PI_4;
    let p = p * f64::cos(xn) - w * q * f64::sin(xn);
    return p * SQRT_FRAC_2_PI / f64::sqrt(x);
}


/// Bessel function of second kind, order zero
///
/// The domain is divided into the intervals [0, 5] and
/// (5, infinity). In the first interval a rational approximation
/// R(x) is employed to compute
/// ```text
///   y0(x) = R(x) + 2 * log(x) * j0(x) / NPY_PI.
/// ```
/// Thus a call to `bessel_j0()` is required.
///
/// In the second interval, the Hankel asymptotic expansion
/// is employed with two rational functions of degree 6/6
/// and 7/7.
///
/// # Accuracy
///
/// ```text
///  Absolute error, when y0(x) < 1; else relative error:
///
/// arithmetic   domain     # trials      peak         rms
///    IEEE      0, 30       30000       1.3e-15     1.6e-16
/// ```
#[allow(clippy::many_single_char_names)]
pub fn bessel_y0(x: f64) -> f64 {
    // Rational approximation coefficients YP[], YQ[] are used here. The
    // function computed is  y0(x) - 2 * log(x) * j0(x) / PI, whose value at
    // x = 0 is  2 * ( log(0.5) + EUL ) / PI = 0.073804295108687225.

    if x == 0.0 {
        // sf_error(SF_ERROR_SINGULAR);
        return -std::f64::INFINITY;
    } else if x < 0.0 {
        // sf_error(SF_ERROR_DOMAIN);
        return std::f64::NAN;
    }

    if x <= 5.0 {
        let z = x * x;
        let mut w = eval_polynomial(z, &YP) / eval_polynomial_1(z, &YQ);
        w += 2.0 / PI * f64::ln(x) * bessel_j0(x);
        return w;
    }

    let w = 5.0 / x;
    let z = 25.0 / (x * x);
    let p = eval_polynomial(z, &PP) / eval_polynomial(z, &PQ);
    let q = eval_polynomial(z, &QP) / eval_polynomial_1(z, &QQ);
    let xn = x - FRAC_PI_4;
    let p = p * f64::sin(xn) + w * q * f64::cos(xn);
    return p * SQRT_FRAC_2_PI / f64::sqrt(x);
}


// Note: all coefficients satisfy the relative error criterion
// except YP, YQ which are designed for absolute error.

static RP: [f64; 4] = [
    -4.794432209782018e9,
    1.9561749194655657e12,
    -2.4924834436096772e14,
    9.708622510473064e15,
];
static RQ: [f64; 8] = [
    4.99563147152651e2,
    1.737854016763747e5,
    4.844096583399621e7,
    1.1185553704535683e10,
    2.112775201154892e12,
    3.1051822985742256e14,
    3.1812195594320496e16,
    1.7108629408104315e18,
];

static PP: [f64; 7] = [
    7.969367292973471e-4,
    8.283523921074408e-2,
    1.239533716464143,
    5.447250030587687,
    8.74716500199817,
    5.303240382353949,
    1.0,
];

static PQ: [f64; 7] = [
    9.244088105588637e-4,
    8.562884743544745e-2,
    1.2535274390105895,
    5.470977403304171,
    8.761908832370695,
    5.306052882353947,
    1.0,
];

static QP: [f64; 8] = [
    -1.1366383889846916e-2,
    -1.2825271867050931,
    -1.9553954425773597e1,
    -9.320601521237683e1,
    -1.7768116798048806e2,
    -1.4707750515495118e2,
    -5.141053267665993e1,
    -6.050143506007285,
];

static QQ: [f64; 7] = [
    6.43178256118178e1,
    8.564300259769806e2,
    3.8824018360540163e3,
    7.240467741956525e3,
    5.930727011873169e3,
    2.0620933166032783e3,
    2.420057402402914e2,
];

static YP: [f64; 8] = [
    1.5592436785523574e4,
    -1.466392959039716e7,
    5.435264770518765e9,
    -9.821360657179115e11,
    8.75906394395367e13,
    -3.466283033847297e15,
    4.4273326857256984e16,
    -1.8495080043698668e16,
];

static YQ: [f64; 7] = [
    1.0412835366425984e3,
    6.26107330137135e5,
    2.6891963339381415e8,
    8.64002487103935e10,
    2.0297961275010555e13,
    3.1715775284297505e15,
    2.5059625617265306e17,
];


#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_j0() {
        // reference computed with mpmath
        assert_relative_eq!(bessel_j0(-123.985), -0.055876145864746804);
        assert_relative_eq!(bessel_j0(-5.1), -0.14433474706050065);
        assert_relative_eq!(bessel_j0(-3.0), -0.26005195490193345);
        assert_relative_eq!(bessel_j0(0.0), 1.0);
        assert_relative_eq!(bessel_j0(0.00245), 0.999998499375563);
        assert_relative_eq!(bessel_j0(2.1752), 0.12419296628748941);
        assert_relative_eq!(bessel_j0(2345.13), 0.012425605700760064, max_relative=1e-12);
    }

    #[test]
    fn test_y0() {
        // reference computed with mpmath
        assert!(bessel_y0(-123.985).is_nan());
        assert!(bessel_y0(-5.1).is_nan());
        assert!(bessel_y0(-3.0).is_nan());
        assert_relative_eq!(bessel_y0(0.0), -f64::INFINITY);
        assert_relative_eq!(bessel_y0(0.00245), -3.90094372498198, max_relative=1e-12);
        assert_relative_eq!(bessel_y0(2.1752), 0.520660638047155, max_relative=1e-12);
        assert_relative_eq!(bessel_y0(2345.13), 0.0108198389384562, max_relative=1e-12);
    }
}
