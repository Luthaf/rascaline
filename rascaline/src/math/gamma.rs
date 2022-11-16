#![allow(clippy::excessive_precision)]
use std::f64;

use approx::ulps_eq;

/// Constant value for `ln(pi)`
pub const LN_PI: f64 = 1.14472988584940017414;

/// Constant value for `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.860382734205265717;

/// Constant value for `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.62078223763524522234;

/// Polynomial coefficients for approximating the `gamma` function
const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

/// Auxiliary variable when evaluating the `gamma` function
const GAMMA_R: f64 = 10.900511;

/// Computes the gamma function with an accuracy of 16 floating point digits.
/// The implementation is derived from "An Analysis of the Lanczos Gamma
/// Approximation", Glendon Ralph Pugh, 2004 p. 116.
///
/// This code was extracted from statrs, which is © 2016 Michael Ma, distributed
/// under MIT license. Cf <https://github.com/boxtown/statrs/blob/c5536a8c916852259832b2064a9b845b68751c8f/src/function/gamma.rs>
/// for the original code
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
///
///
/// This code was extracted from statrs, which is © 2016 Michael Ma, distributed
/// under MIT license. Cf <https://github.com/boxtown/statrs/blob/c5536a8c916852259832b2064a9b845b68751c8f/src/function/gamma.rs>
/// for the original code
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
    let d1 = -0.57721566490153286;
    let d2 = 1.6449340668482264365;
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
        assert_relative_eq!(gamma(1.000001e-35), 9.9999900000099999900000099999899999522784235098567139293e+34, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-10), 9.99998999943278432519738283781280989934496494539074049002e+9, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-5), 99999.32279432557746387178953902739303931424932435387031653234, max_relative=1e-13);
        assert_relative_eq!(gamma(1.000001e-2), 99.43248512896257405886134437203369035261893114349805309870831, max_relative=1e-13);
        assert_relative_eq!(gamma(-4.8), -0.06242336135475955314181664931547009890495158793105543559676, max_relative=1e-13);
        assert_relative_eq!(gamma(-1.5), 2.363271801207354703064223311121526910396732608163182837618410, max_relative=1e-13);
        assert_relative_eq!(gamma(-0.5), -3.54490770181103205459633496668229036559509891224477425642761, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0e-5 + 1.0e-16), 99999.42279322556767360213300482199406241771308740302819426480, max_relative=1e-13);
        assert_relative_eq!(gamma(0.1), 9.513507698668731836292487177265402192550578626088377343050000, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0 - 1.0e-14), 1.000000000000005772156649015427511664653698987042926067639529, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0), 1.0, max_relative=1e-13);
        assert_relative_eq!(gamma(1.0 + 1.0e-14), 0.99999999999999422784335098477029953441189552403615306268023, max_relative=1e-13);
        assert_relative_eq!(gamma(1.5), 0.886226925452758013649083741670572591398774728061193564106903, max_relative=1e-13);
        assert_relative_eq!(gamma(std::f64::consts::PI / 2.0), 0.890560890381539328010659635359121005933541962884758999762766, max_relative=1e-13);
        assert_relative_eq!(gamma(2.0), 1.0, max_relative=1e-13);
        assert_relative_eq!(gamma(2.5), 1.329340388179137020473625612505858887098162092091790346160355, max_relative=1e-13);
        assert_relative_eq!(gamma(3.0), 2.0, max_relative=1e-13);
        assert_relative_eq!(gamma(std::f64::consts::PI), 2.288037795340032417959588909060233922889688153356222441199380, max_relative=1e-13);
        assert_relative_eq!(gamma(3.5), 3.323350970447842551184064031264647217745405230229475865400889, max_relative=1e-13);
        assert_relative_eq!(gamma(4.0), 6.0, max_relative=1e-13);
        assert_relative_eq!(gamma(4.5), 11.63172839656744892914422410942626526210891830580316552890311, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0 - 1.0e-14), 23.99999999999963853175957637087420162718107213574617032780374, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0), 24.0, max_relative=1e-13);
        assert_relative_eq!(gamma(5.0 + 1.0e-14), 24.00000000000036146824042363510111050137786752408660789873592, max_relative=1e-13);
        assert_relative_eq!(gamma(5.5), 52.34277778455352018114900849241819367949013237611424488006401, max_relative=1e-13);
        assert_relative_eq!(gamma(10.1), 454760.7514415859508673358368319076190405047458218916492282448, max_relative=1e-13);
        assert_relative_eq!(gamma(150.0 + 1.0e-12), 3.8089226376496421386707466577615064443807882167327097140e+260, max_relative=1e-12);
    }

    #[test]
    fn test_ln_gamma() {
        assert!(ln_gamma(f64::NAN).is_nan());
        assert_eq!(ln_gamma(1.000001e-35), 80.59047725479209894029636783061921392709972287131139201585211);
        assert_relative_eq!(ln_gamma(1.000001e-10), 23.02584992988323521564308637407936081168344192865285883337793, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.000001e-5), 11.51291869289055371493077240324332039045238086972508869965363, max_relative=1e-13);
        assert_eq!(ln_gamma(1.000001e-2), 4.599478872433667224554543378460164306444416156144779542513592);
        assert_relative_eq!(ln_gamma(0.1), 2.252712651734205959869701646368495118615627222294953765041739, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.0 - 1.0e-14), 5.772156649015410852768463312546533565566459794933360600e-15, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.0), 0.0, epsilon=1e-15);
        assert_relative_eq!(ln_gamma(1.0 + 1.0e-14), -5.77215664901524635936177848990288632404978978079827014e-15, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(1.5), -0.12078223763524522234551844578164721225185272790259946836386, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(f64::consts::PI/2.0), -0.11590380084550241329912089415904874214542604767006895, max_relative=1e-13);
        assert_eq!(ln_gamma(2.0), 0.0);
        assert_relative_eq!(ln_gamma(2.5), 0.284682870472919159632494669682701924320137695559894729250145, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(3.0), 0.693147180559945309417232121458176568075500134360255254120680, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(f64::consts::PI), 0.82769459232343710152957855845235995115350173412073715, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(3.5), 1.200973602347074224816021881450712995770238915468157197042113, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(4.0), 1.791759469228055000812477358380702272722990692183004705855374, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(4.5), 2.453736570842442220504142503435716157331823510689763131380823, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0 - 1.0e-14), 3.178053830347930558470257283303394288448414225994179545985931, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0), 3.178053830347945619646941601297055408873990960903515214096734, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.0 + 1.0e-14), 3.178053830347960680823625919312848824873279228348981287761046, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(5.5), 3.957813967618716293877400855822590998551304491975006780729532, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(10.1), 13.02752673863323795851370097886835481188051062306253294740504, max_relative=1e-13);
        assert_relative_eq!(ln_gamma(150.0 + 1.0e-12), 600.0094705553324354062157737572509902987070089159051628001813, max_relative=1e-12);
        assert_relative_eq!(ln_gamma(1.001e+7), 1.51342135323817913130119829455205139905331697084416059779e+8, max_relative=1e-13);
    }

    #[test]
    fn test_digamma() {
        use std::f64::consts::{FRAC_PI_2, LN_2};

        assert!(digamma(f64::NAN).is_nan());
        assert_relative_eq!(digamma(-1.5), 0.70315664064524318722569033366791109947350706200623256, max_relative=1e-13);
        assert_relative_eq!(digamma(-0.5), 0.036489973978576520559023667001244432806840395339565891, max_relative=1e-13);
        assert_relative_eq!(digamma(0.1), -10.423754940411076232100295314502760886768558023951363, max_relative=1e-13);
        assert_relative_eq!(digamma(0.25), -FRAC_PI_2 - 3.0 * LN_2 - EULER, max_relative=1e-13);
        assert_relative_eq!(digamma(1.0), -EULER, max_relative=1e-13);
        assert_relative_eq!(digamma(1.5), 0.036489973978576520559023667001244432806840395339565888, max_relative=1e-13);
        assert_relative_eq!(digamma(f64::consts::PI / 2.0), 0.10067337642740238636795561404029690452798358068944001, max_relative=1e-13);
        assert_relative_eq!(digamma(2.0), 0.42278433509846713939348790991759756895784066406007641, max_relative=1e-13);
        assert_relative_eq!(digamma(2.5), 0.70315664064524318722569033366791109947350706200623255, max_relative=1e-13);
        assert_relative_eq!(digamma(3.0), 0.92278433509846713939348790991759756895784066406007641, max_relative=1e-13);
        assert_relative_eq!(digamma(f64::consts::PI), 0.97721330794200673329206948640618234364083460999432603, max_relative=1e-13);
        assert_relative_eq!(digamma(3.5), 1.1031566406452431872256903336679110994735070620062326, max_relative=1e-13);
        assert_relative_eq!(digamma(4.0), 1.2561176684318004727268212432509309022911739973934097, max_relative=1e-13);
        assert_relative_eq!(digamma(4.5), 1.3888709263595289015114046193821968137592213477205183, max_relative=1e-13);
        assert_relative_eq!(digamma(5.0), 1.5061176684318004727268212432509309022911739973934097, max_relative=1e-13);
        assert_relative_eq!(digamma(5.5), 1.6110931485817511237336268416044190359814435699427405, max_relative=1e-13);
        assert_relative_eq!(digamma(10.1), 2.2622143570941481235561593642219403924532310597356171, max_relative=1e-13);
    }
}
