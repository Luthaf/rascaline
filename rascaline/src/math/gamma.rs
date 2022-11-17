//! This code was extracted from statrs, which is © 2016 Michael Ma, distributed
//! under MIT license.
//!
//! See <https://github.com/boxtown/statrs/blob/c5536a8c/src/function/gamma.rs>
//! for the original code.

use std::f64;

use approx::ulps_eq;

/// Constant value for `ln(pi)`
pub const LN_PI: f64 = 1.1447298858494002;

/// Constant value for `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657;

/// Constant value for `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452;

/// Polynomial coefficients for approximating the `gamma` function
const GAMMA_DK: &[f64] = &[
    2.4857408913875355e-5,
    1.0514237858172197,
    -3.4568709722201625,
    4.512277094668948,
    -2.9828522532357664,
    1.056397115771267,
    -1.9542877319164587e-1,
    1.709705434044412e-2,
    -5.719261174043057e-4,
    4.633994733599057e-6,
    -2.7199490848860772e-9,
];

/// Auxiliary variable when evaluating the `gamma` function
const GAMMA_R: f64 = 10.900511;

/// Computes the gamma function with an accuracy of 16 floating point digits.
/// The implementation is derived from "An Analysis of the Lanczos Gamma
/// Approximation", Glendon Ralph Pugh, 2004 p. 116.
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        std::f64::consts::PI
            / ((std::f64::consts::PI * x).sin()
                * s
                * TWO_SQRT_E_OVER_PI
                * ((0.5 - x + GAMMA_R) / std::f64::consts::E).powf(0.5 - x))
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s * TWO_SQRT_E_OVER_PI * ((x - 0.5 + GAMMA_R) / std::f64::consts::E).powf(x - 0.5)
    }
}

/// Computes the logarithm of the gamma function with an accuracy of 16 floating
/// point digits. The implementation is derived from "An Analysis of the Lanczos
/// Gamma Approximation", Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        LN_PI
            - (f64::consts::PI * x).sin().ln()
            - s.ln()
            - LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s.ln()
            + LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
    }
}

/// Computes the upper incomplete gamma function `Gamma(a,x) =
/// int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0` where `a` is the argument for
/// the gamma function and `x` is the lower integral limit.
#[allow(dead_code)]
pub fn gamma_ui(a: f64, x: f64) -> f64 {
    return gamma_ur(a, x) * gamma(a);
}

/// Computes the upper incomplete regularized gamma function `Q(a,x) = 1 /
/// Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0` where `a` is the
/// argument for the gamma function and `x` is the lower integral limit.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
#[allow(dead_code)]
pub fn gamma_ur(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }

    if a <= 0.0 || a == f64::INFINITY {
        panic!("invalid argument a to gamma_ur");
    }

    if x <= 0.0 || x == f64::INFINITY {
        panic!("invalid argument x to gamma_ur");
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.220446049250313e-16;

    if x < 1.0 || x <= a {
        return 1.0 - gamma_lr(a, x);
    }

    let mut ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.782712893384 {
        return if a < x { 0.0 } else { 1.0 };
    }

    ax = ax.exp();
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    loop {
        y += 1.0;
        z += 2.0;
        c += 1.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if pk.abs() > big {
            pkm2 *= big_inv;
            pkm1 *= big_inv;
            qkm2 *= big_inv;
            qkm1 *= big_inv;
        }

        if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;

            if t <= eps {
                break;
            }
        }
    }

    return ans * ax;
}

/// Computes the lower incomplete regularized gamma function `P(a,x) = 1 /
/// Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for real a > 0, x > 0` where `a` is
/// the argument for the gamma function and `x` is the upper integral limit.
#[allow(clippy::many_single_char_names)]
#[allow(dead_code)]
pub fn gamma_lr(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || a == f64::INFINITY {
        panic!("invalid argument a to gamma_lr");
    }
    if x <= 0.0 || x == f64::INFINITY {
        panic!("invalid argument x to gamma_lr");
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.220446049250313e-16;

    if a == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    let ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.782712893384 {
        if a < x {
            return 1.0;
        }
        return 0.0;
    }

    if x <= 1.0 || x <= a {
        let mut r2 = a;
        let mut c2 = 1.0;
        let mut ans2 = 1.0;
        loop {
            r2 += 1.0;
            c2 *= x / r2;
            ans2 += c2;

            if c2 / ans2 <= eps {
                break;
            }
        }
        return ax.exp() * ans2 / a;
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0;

    let mut p3 = 1.0;
    let mut q3 = x;
    let mut p2 = x + 1.0;
    let mut q2 = z * x;
    let mut ans = p2 / q2;

    loop {
        y += 1.0;
        z += 2.0;
        c += 1;
        let yc = y * f64::from(c);

        let p = p2 * z - p3 * yc;
        let q = q2 * z - q3 * yc;

        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        if p.abs() > big {
            p3 *= big_inv;
            p2 *= big_inv;
            q3 *= big_inv;
            q2 *= big_inv;
        }

        if q != 0.0 {
            let nextans = p / q;
            let error = ((ans - nextans) / nextans).abs();
            ans = nextans;

            if error <= eps {
                break;
            }
        }
    }

    return 1.0 - ax.exp() * ans;
}

/// Computes the Digamma function which is defined as the derivative of
/// the log of the gamma function. The implementation is based on
/// "Algorithm AS 103", Jose Bernardo, Applied Statistics, Volume 25, Number 3
/// 1976, pages 315 - 317
///
///
/// This code was extracted from statrs, which is © 2016 Michael Ma, distributed
/// under MIT license. Cf <https://github.com/statrs-dev/statrs/blob/5411ba74b427b992dd1410d699052a0b41dc2b5c/src/function/gamma.rs>
/// for the original code
#[allow(clippy::many_single_char_names)]
pub fn digamma(x: f64) -> f64 {
    let c = 12.0;
    let d1 = -0.5772156649015329;
    let d2 = 1.6449340668482264;
    let s = 1e-6;
    let s3 = 1.0 / 12.0;
    let s4 = 1.0 / 120.0;
    let s5 = 1.0 / 252.0;
    let s6 = 1.0 / 240.0;
    let s7 = 1.0 / 132.0;

    if x == f64::NEG_INFINITY || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && ulps_eq!(x.floor(), x) {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return digamma(1.0 - x) + f64::consts::PI / (-f64::consts::PI * x).tan();
    }
    if x <= s {
        return d1 - 1.0 / x + d2 * x;
    }

    let mut result = 0.0;
    let mut z = x;
    while z < c {
        result -= 1.0 / z;
        z += 1.0;
    }

    if z >= c {
        let mut r = 1.0 / z;
        result += z.ln() - 0.5 * r;
        r *= r;

        result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::math::EULER;

    #[test]
    fn test_gamma() {
        assert!(gamma(f64::NAN).is_nan());
        assert_relative_eq!(gamma(1.000001e-35), 9.99999000001e34, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-10), 9.999989999432785e9, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-5), 99999.32279432558, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-2), 99.43248512896257, max_relative=1e-13);
        assert_relative_eq!(gamma(-4.8), -0.06242336135475955, max_relative=1e-13);
        assert_relative_eq!(gamma(-1.5), 2.363271801207355, max_relative=1e-13);
        assert_relative_eq!(gamma(-0.5), -3.544907701811032, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0e-5 + 1.0e-16), 99999.42279322557, max_relative=1e-13);
        assert_relative_eq!(gamma(0.1), 9.513507698668732, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0 - 1.0e-14), 1.0000000000000058, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0), 1.0, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0 + 1.0e-14), 0.9999999999999942, max_relative=1e-13);
        assert_relative_eq!(gamma(1.5), 0.886226925452758, max_relative=1e-13);
        assert_relative_eq!(gamma(std::f64::consts::PI / 2.0), 0.8905608903815393, max_relative=1e-13);
        assert_relative_eq!(gamma(2.0), 1.0, max_relative=1e-13);
        assert_relative_eq!(gamma(2.5), 1.329340388179137, max_relative=1e-13);
        assert_relative_eq!(gamma(3.0), 2.0, max_relative=1e-13);
        assert_relative_eq!(gamma(std::f64::consts::PI), 2.2880377953400326, max_relative=1e-13);
        assert_relative_eq!(gamma(3.5), 3.3233509704478426, max_relative=1e-13);
        assert_relative_eq!(gamma(4.0), 6.0, max_relative=1e-13);
        assert_relative_eq!(gamma(4.5), 11.631728396567448, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0 - 1.0e-14), 23.999999999999638, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0), 24.0, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0 + 1.0e-14), 24.000000000000362, max_relative=1e-13);
        assert_relative_eq!(gamma(5.5), 52.34277778455352, max_relative=1e-13);
        assert_relative_eq!(gamma(10.1), 454760.75144158595, max_relative=1e-13);
        assert_relative_eq!(gamma(150.0 + 1.0e-12), 3.808922637649642e260, max_relative=1e-12);
    }

    #[test]
    fn test_ln_gamma() {
        assert!(ln_gamma(f64::NAN).is_nan());
        assert_eq!(ln_gamma(1.000001e-35), 80.5904772547921);
        assert_relative_eq!(ln_gamma(1.000001e-10), 23.025849929883236, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.000001e-5), 11.512918692890553, max_relative=1e-13);
        assert_eq!(ln_gamma(1.000001e-2), 4.599478872433667);
        assert_relative_eq!(ln_gamma(0.1), 2.252712651734206, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.0 - 1.0e-14), 5.772156649015411e-15, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon=1e-15);
        assert_relative_eq!(ln_gamma(1.0 + 1.0e-14), -5.772156649015246e-15, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.5), -0.12078223763524522, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(f64::consts::PI/2.0), -0.11590380084550242, max_relative=1e-13);
        assert_eq!(ln_gamma(2.0), 0.0);
        assert_relative_eq!(ln_gamma(2.5), 0.2846828704729192, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(3.0), std::f64::consts::LN_2, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(f64::consts::PI), 0.8276945923234371, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(3.5), 1.2009736023470743, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(4.0), 1.791759469228055, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(4.5), 2.4537365708424423, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0 - 1.0e-14), 3.1780538303479307, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0), 3.1780538303479458, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0 + 1.0e-14), 3.178053830347961, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.5), 3.9578139676187165, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(10.1), 13.027526738633238, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(150.0 + 1.0e-12), 600.0094705553324, max_relative=1e-12);
        assert_relative_eq!(ln_gamma(1.001e+7), 1.513421353238179e8, max_relative=1e-13);
    }

    #[test]
    fn test_gamma_lr() {
        assert!(gamma_lr(f64::NAN, f64::NAN).is_nan());
        assert_relative_eq!(gamma_lr(0.1, 1.0), 0.9758726562736723, max_relative=1e-14);
        assert_eq!(gamma_lr(0.1, 2.0), 0.9943261760201885);
        assert_eq!(gamma_lr(0.1, 8.0), 0.999995075192052);
        assert_relative_eq!(gamma_lr(1.5, 1.0), 0.4275932955291202, max_relative=1e-13);
        assert_relative_eq!(gamma_lr(1.5, 2.0), 0.7385358700508894, max_relative=1e-15);
        assert_eq!(gamma_lr(1.5, 8.0), 0.9988660157102147);
        assert_relative_eq!(gamma_lr(2.5, 1.0), 0.15085496391539036, max_relative=1e-13);
        assert_relative_eq!(gamma_lr(2.5, 2.0), 0.4505840486472198, max_relative=1e-14);
        assert_relative_eq!(gamma_lr(2.5, 8.0), 0.9931559260775795, max_relative=1e-15);
        assert_relative_eq!(gamma_lr(5.5, 1.0), 0.0015041182825838038, max_relative=1e-15);
        assert_relative_eq!(gamma_lr(5.5, 2.0), 0.03008297612122605, max_relative=1e-14);
        assert_relative_eq!(gamma_lr(5.5, 8.0), 0.8588691197329419, max_relative=1e-14);
        assert_relative_eq!(gamma_lr(100.0, 0.5), 0.0);
        assert_relative_eq!(gamma_lr(100.0, 1.5), 0.0);
        assert_relative_eq!(gamma_lr(100.0, 90.0), 0.15822098918643016, max_relative=1e-9);
        assert_relative_eq!(gamma_lr(100.0, 100.0), 0.5132987982791487, max_relative=1e-13);
        assert_relative_eq!(gamma_lr(100.0, 110.0), 0.8417213299399129, max_relative=1e-13);
        assert_relative_eq!(gamma_lr(100.0, 200.0), 1.0, max_relative=1e-14);
        assert_eq!(gamma_lr(500.0, 0.5), 0.0);
        assert_eq!(gamma_lr(500.0, 1.5), 0.0);
        assert_relative_eq!(gamma_lr(500.0, 200.0), 0.0, max_relative=1e-70);
        assert_relative_eq!(gamma_lr(500.0, 450.0), 0.010717238091289742, max_relative=1e-9);
        assert_relative_eq!(gamma_lr(500.0, 500.0), 0.5059471461707603, max_relative=1e-13);
        assert_relative_eq!(gamma_lr(500.0, 550.0), 0.9853855918737048, max_relative=1e-14);
        assert_relative_eq!(gamma_lr(500.0, 700.0), 1.0, max_relative=1e-15);
        assert_eq!(gamma_lr(1000.0, 10000.0), 1.0);
        assert_eq!(gamma_lr(1e+50, 1e+48), 0.0);
        assert_eq!(gamma_lr(1e+50, 1e+52), 1.0);
    }

    #[test]
    fn test_gamma_ur() {
        assert!(gamma_ur(f64::NAN, f64::NAN).is_nan());
        assert_relative_eq!(gamma_ur(0.1, 1.0), 0.024127343726327778, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(0.1, 2.0), 0.005673823979811528, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(0.1, 8.0), 0.000004924807948019513, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(1.5, 1.0), 0.5724067044708798, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(1.5, 2.0), 0.2614641299491106, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(1.5, 8.0), 0.0011339842897853227, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(2.5, 1.0), 0.8491450360846097, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(2.5, 2.0), 0.5494159513527802, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(2.5, 8.0), 0.006844073922420431, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(5.5, 1.0), 0.9984958817174162, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(5.5, 2.0), 0.969917023878774, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(5.5, 8.0), 0.14113088026705814, max_relative=1e-13);
        assert_relative_eq!(gamma_ur(100.0, 0.5), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(100.0, 1.5), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(100.0, 90.0), 0.8417790108135699, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(100.0, 100.0), 0.48670120172085135, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(100.0, 110.0), 0.15827867006008708, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(100.0, 200.0), 0.0, epsilon=1e-14);
        assert_relative_eq!(gamma_ur(500.0, 0.5), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(500.0, 1.5), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(500.0, 200.0), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(500.0, 450.0), 0.9892827619087102, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(500.0, 500.0), 0.49405285382923964, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(500.0, 550.0), 0.014614408126295194, max_relative=1e-12);
        assert_relative_eq!(gamma_ur(500.0, 700.0), 0.0, epsilon=1e-14);
        assert_relative_eq!(gamma_ur(1000.0, 10000.0), 0.0, epsilon=1e-14);
        assert_relative_eq!(gamma_ur(1e+50, 1e+48), 1.0, max_relative=1e-14);
        assert_relative_eq!(gamma_ur(1e+50, 1e+52), 0.0, epsilon=1e-14);
    }

    #[test]
    fn test_gamma_ui() {
        assert!(gamma_ui(f64::NAN, f64::NAN).is_nan());
        assert_relative_eq!(gamma_ui(0.1, 1.0), 0.22953567028884603, max_relative=1e-14);
        assert_relative_eq!(gamma_ui(0.1, 2.0), 0.053977968112828234, max_relative=1e-15);
        assert_relative_eq!(gamma_ui(0.1, 8.0), 0.00004685219832794859, max_relative=1e-19);
        assert_relative_eq!(gamma_ui(1.5, 1.0), 0.5072822338117733, max_relative=1e-14);
        assert_relative_eq!(gamma_ui(1.5, 2.0), 0.23171655200098068, max_relative=1e-15);
        assert_relative_eq!(gamma_ui(1.5, 8.0), 0.001004967410648176, max_relative=1e-17);
        assert_relative_eq!(gamma_ui(2.5, 1.0), 1.1288027918891024, max_relative=1e-14);
        assert_relative_eq!(gamma_ui(2.5, 2.0), 0.7303608140431147, max_relative=1e-14);
        assert_relative_eq!(gamma_ui(2.5, 8.0), 0.009098103884757085, max_relative=1e-17);
        assert_relative_eq!(gamma_ui(5.5, 1.0), 52.26404805552655, max_relative=1e-12);
        assert_relative_eq!(gamma_ui(5.5, 2.0), 50.76815125034216, max_relative=1e-12);
        assert_relative_eq!(gamma_ui(5.5, 8.0), 7.387182304357054, max_relative=1e-13);
    }

    #[test]
    fn test_digamma() {
        use std::f64::consts::{FRAC_PI_2, LN_2};

        assert!(digamma(f64::NAN).is_nan());
        assert_relative_eq!(digamma(-1.5), 0.7031566406452432, max_relative=1e-13);
        assert_relative_eq!(digamma(-0.5), 0.03648997397857652, max_relative=1e-13);
        assert_relative_eq!(digamma(0.1), -10.423754940411076, max_relative=1e-13);
        assert_relative_eq!(digamma(0.25), -FRAC_PI_2 - 3.0 * LN_2 - EULER, max_relative=1e-13);
        assert_relative_eq!(digamma(1.0), -EULER, max_relative=1e-13);
        assert_relative_eq!(digamma(1.5), 0.03648997397857652, max_relative=1e-13);
        assert_relative_eq!(digamma(f64::consts::PI / 2.0), 0.10067337642740239, max_relative=1e-13);
        assert_relative_eq!(digamma(2.0), 0.42278433509846713, max_relative=1e-13);
        assert_relative_eq!(digamma(2.5), 0.7031566406452432, max_relative=1e-13);
        assert_relative_eq!(digamma(3.0), 0.9227843350984671, max_relative=1e-13);
        assert_relative_eq!(digamma(f64::consts::PI), 0.9772133079420068, max_relative=1e-13);
        assert_relative_eq!(digamma(3.5), 1.103156640645243, max_relative=1e-13);
        assert_relative_eq!(digamma(4.0), 1.2561176684318005, max_relative=1e-13);
        assert_relative_eq!(digamma(4.5), 1.388870926359529, max_relative=1e-13);
        assert_relative_eq!(digamma(5.0), 1.5061176684318005, max_relative=1e-13);
        assert_relative_eq!(digamma(5.5), 1.6110931485817512, max_relative=1e-13);
        assert_relative_eq!(digamma(10.1), 2.262214357094148, max_relative=1e-13);
    }
}
