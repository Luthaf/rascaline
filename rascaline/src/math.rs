#![allow(clippy::excessive_precision)]

/// Auxiliary variable when evaluating the `gamma` function
const GAMMA_R: f64 = 10.900511;

/// Constant value for `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.860382734205265717;

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

/// Computes the gamma function with an accuracy of 16 floating point digits.
/// The implementation is derived from "An Analysis of the Lanczos Gamma
/// Approximation", Glendon Ralph Pugh, 2004 p. 116.
///
/// This code was extracted from statrs, which is © 2016 Michael Ma, distributed
/// under MIT license. Cf https://github.com/boxtown/statrs/blob/c5536a8c916852259832b2064a9b845b68751c8f/src/function/gamma.rs
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gamma() {
        assert!(gamma(f64::NAN).is_nan());
        assert_relative_eq!(gamma(1.000001e-35), 9.9999900000099999900000099999899999522784235098567139293e+34, max_relative=1e-13); //, 1e20);
        assert_relative_eq!(gamma(1.000001e-10), 9.99998999943278432519738283781280989934496494539074049002e+9, max_relative=1e-13); //, 1e-5);
        assert_relative_eq!(gamma(1.000001e-5), 99999.32279432557746387178953902739303931424932435387031653234, max_relative=1e-13); //, 1e-10);
        assert_relative_eq!(gamma(1.000001e-2), 99.43248512896257405886134437203369035261893114349805309870831, max_relative=1e-13); //, 1e-13);
        assert_relative_eq!(gamma(-4.8), -0.06242336135475955314181664931547009890495158793105543559676, max_relative=1e-13); //, 1e-13);
        assert_relative_eq!(gamma(-1.5), 2.363271801207354703064223311121526910396732608163182837618410, max_relative=1e-13); //, 1e-13);
        assert_relative_eq!(gamma(-0.5), -3.54490770181103205459633496668229036559509891224477425642761, max_relative=1e-13); //, 1e-13);
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
}
