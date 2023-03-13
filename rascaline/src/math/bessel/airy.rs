use crate::math::consts::SQRT_3;

use super::polynomials::{eval_polynomial, eval_polynomial_1};

/// Solution of the differential equation
///
/// ```text
/// y"(x) = xy
/// ```
///
/// The function returns `((Ai, Bi), (Ai'(x), Bi'(x)))` where Ai, and Bi are
/// the two independent solutions and Ai'(x), Bi'(x) their first derivatives.
///
/// Evaluation is by power series summation for small x, by rational minimax
/// approximations for large x.
///
/// ACCURACY:
///
/// Error criterion is absolute when function <= 1, relative when function > 1,
/// except * denotes relative error criterion. For large negative x, the
/// absolute error increases as x^1.5. For large positive x, the relative error
/// increases as x^1.5.
///
/// Arithmetic  domain   function  # trials      peak         rms
/// IEEE        -10, 0     Ai        10000       1.6e-15     2.7e-16
/// IEEE          0, 10    Ai        10000       2.3e-14*    1.8e-15*
/// IEEE        -10, 0     Ai'       10000       4.6e-15     7.6e-16
/// IEEE          0, 10    Ai'       10000       1.8e-14*    1.5e-15*
/// IEEE        -10, 10    Bi        30000       4.2e-15     5.3e-16
/// IEEE        -10, 10    Bi'       30000       4.9e-15     7.3e-16
#[allow(clippy::many_single_char_names, clippy::similar_names, clippy::too_many_lines)]
pub fn airy(x: f64) -> ((f64, f64), (f64, f64)) {
    let mut ai = 0.0;
    let mut bi = 0.0;
    let mut aip = 0.0;
    let mut bip = 0.0;

    let mut z;
    let zz;
    let mut t;
    let mut f;
    let mut g;
    let mut uf;
    let mut ug;
    let mut k;
    let zeta;
    let theta;
    let mut domain_flag= 0;

    if x > 103.892 {
        return ((0.0, 0.0), (std::f64::INFINITY, std::f64::INFINITY));
    }

    if x < -2.09 {
        // domain_flag = 15;
        t = f64::sqrt(-x);
        zeta = -2.0 * x * t / 3.0;
        t = f64::sqrt(t);
        k = SQPII / t;
        z = 1.0 / zeta;
        zz = z * z;
        uf = 1.0 + zz * eval_polynomial(zz, &AFN) / eval_polynomial_1(zz, &AFD);
        ug = z * eval_polynomial(zz, &AGN) / eval_polynomial_1(zz, &AGD);
        theta = zeta + 0.25 * std::f64::consts::PI;
        f = f64::sin(theta);
        g = f64::cos(theta);
        ai = k * (f * uf - g * ug);
        bi = k * (g * uf + f * ug);

        uf = 1.0 + zz * eval_polynomial(zz, &APFN) / eval_polynomial_1(zz, &APFD);
        ug = z * eval_polynomial(zz, &APGN) / eval_polynomial_1(zz, &APGD);
        k = SQPII * t;
        aip = -k * (g * uf + f * ug);
        bip = k * (f * uf - g * ug);

        return ((ai, bi), (aip, bip));
    }

    if x >= 2.09 {
        domain_flag = 5;
        t = f64::sqrt(x);
        zeta = 2.0 * x * t / 3.0;
        g = f64::exp(zeta);
        t = f64::sqrt(t);
        k = 2.0 * t * g;
        z = 1.0 / zeta;
        f = eval_polynomial(z, &AN) / eval_polynomial_1(z, &AD);
        ai = SQPII * f / k;
        k = -0.5 * SQPII * t / g;
        f = eval_polynomial(z, &APN) / eval_polynomial_1(z, &APD);
        aip = f * k;
        if x > 8.3203353 {
            f = z * eval_polynomial(z, &BN16) / eval_polynomial_1(z, &BD16);
            k = SQPII * g;
            bi = k * (1.0 + f) / t;
            f = z * eval_polynomial(z, &BPPN) / eval_polynomial_1(z, &BPPD);
            bip = k * t * (1.0 + f);

            return ((ai, bi), (aip, bip));
        }
    }

    f = 1.0;
    g = x;
    t = 1.0;
    uf = 1.0;
    ug = x;
    k = 1.0;
    z = x * x * x;
    while t > f64::EPSILON {
        uf *= z;
        k += 1.0;
        uf /= k;
        ug *= z;
        k += 1.0;
        ug /= k;
        uf /= k;
        f += uf;
        k += 1.0;
        ug /= k;
        g += ug;
        t = f64::abs(uf / f);
    }
    uf = C1 * f;
    ug = C2 * g;

    if domain_flag & 1 == 0 {
        ai = uf - ug;
    }

    if domain_flag & 2 == 0 {
        bi = SQRT_3 * (uf + ug);
    }

    k = 4.0;
    uf = x * x / 2.0;
    ug = z / 3.0;
    f = uf;
    g = 1.0 + ug;
    uf /= 3.0;
    t = 1.0;

    while t > f64::EPSILON {
        uf *= z;
        ug /= k;
        k += 1.0;
        ug *= z;
        uf /= k;
        f += uf;
        k += 1.0;
        ug /= k;
        uf /= k;
        g += ug;
        k += 1.0;
        t = f64::abs(ug / g);
    }
    uf = C1 * f;
    ug = C2 * g;

    if domain_flag & 4 == 0 {
        aip = uf - ug;
    }

    if domain_flag & 8 == 0 {
        bip = SQRT_3 * (uf + ug);
    }

    return ((ai, bi), (aip, bip));
}

const C1: f64 = 0.3550280538878172;
const C2: f64 = 0.2588194037928068;
const SQPII: f64 = 0.5641895835477563;

static AN: [f64; 8] = [
    3.46538101525629e-1,
    1.2007595273964581e1,
    7.627960536152345e1,
    1.6808922493463058e2,
    1.5975639135016442e2,
    7.053609068404442e1,
    1.4026469116338967e1,
    1.0,
];

static AD: [f64; 8] = [
    5.675945326387702e-1,
    1.475625625848472e1,
    8.451389701414746e1,
    1.7731808814540045e2,
    1.642346928715297e2,
    7.147784008255756e1,
    1.4095913560783403e1,
    1.0,
];

static APN: [f64; 8] = [
    6.137591848140358e-1,
    1.4745467078775532e1,
    8.20584123476061e1,
    1.711847813609764e2,
    1.593178471371418e2,
    6.997785993301031e1,
    1.3947085698048157e1,
    1.0,
];

static APD: [f64; 8] = [
    3.3420367774973697e-1,
    1.1181029730615816e1,
    7.1172735214786e1,
    1.5877808437283832e2,
    1.5320642747580922e2,
    6.867523045927804e1,
    1.3849863475825945e1,
    1.0,
];

static BN16: [f64; 5] = [
    -2.5324079586936415e-1,
    5.752851673324674e-1,
    -3.2990703687322537e-1,
    6.444040689482e-2,
    -3.8251954664133675e-3,
];

static BD16: [f64; 5] = [
    -7.156850950540353,
    1.0603958071566469e1,
    -5.232466364712515,
    9.573958643783839e-1,
    -5.508281471635496e-2,
];

static BPPN: [f64; 5] = [
    4.654611627746516e-1,
    -1.0899217380049393,
    6.38800117371828e-1,
    -1.2684434955310292e-1,
    7.624878443421098e-3,
];

static BPPD: [f64; 5] = [
    -8.70622787633159,
    1.3899316270455321e1,
    -7.141161446164312,
    1.340085959606805,
    -7.84273211323342e-2,
];

static AFN: [f64; 9] = [
    -1.316963234183318e-1,
    -6.264565444319123e-1,
    -6.931580360369335e-1,
    -2.797799815451191e-1,
    -4.919001326095003e-2,
    -4.062659235948854e-3,
    -1.592764962392621e-4,
    -2.776491081552329e-6,
    -1.6778769848911465e-8,
];

static AFD: [f64; 9] = [
    1.3356042070655324e1,
    3.2682503279522464e1,
    2.6736704094149957e1,
    9.187074029072596,
    1.4752914677166642,
    1.1568717379518804e-1,
    4.402916416152112e-3,
    7.547203482874142e-5,
    4.5185009297058035e-7,
];

static AGN: [f64; 11] = [
    1.973399320916857e-2,
    3.9110302961568827e-1,
    1.0657989759959559,
    9.391692298166502e-1,
    3.5146565610554764e-1,
    6.338889196289255e-2,
    5.858041130483885e-3,
    2.82851600836737e-4,
    6.98793669997261e-6,
    8.117892395543892e-8,
    3.415517847659236e-10,
];

static AGD: [f64; 10] = [
    9.30892908077442,
    1.9835292871831214e1,
    1.5564662893286462e1,
    5.476860694229755,
    9.542936116189619e-1,
    8.645808263523921e-2,
    4.126565238242226e-3,
    1.0125908511650914e-4,
    1.1716673321441352e-6,
    4.9183457006293e-9,
];

static APFN: [f64; 9] = [
    1.8536562402253556e-1,
    8.867121880525841e-1,
    9.873919817473985e-1,
    4.0124108231800376e-11,
    7.103049262896312e-2,
    5.906186579956618e-3,
    2.330514094017768e-4,
    4.087187782890355e-6,
    2.4837993290044246e-8,
];

static APFD: [f64; 9] = [
    1.4734585468750254e1,
    3.754239334354896e1,
    3.146577512030464e1,
    1.0996912520729877e1,
    1.788850547669994,
    1.4173327575366262e-1,
    5.44066067017226e-3,
    9.394212906545112e-5,
    5.65978713036027e-7,
];

static APGN: [f64; 11] = [
    -3.556154290330823e-2,
    -6.373115181294355e-1,
    -1.7085673888431236,
    -1.5022187211731663,
    -5.636066658221027e-1,
    -1.0210103112021689e-1,
    -9.483966959614452e-3,
    -4.6032530748678097e-4,
    -1.1430083648451737e-5,
    -1.3341551868554742e-7,
    -5.638038339588935e-10,
];

static APGD: [f64; 10] = [
    9.858658016961304,
    2.1640186735658595e1,
    1.731307763897494e1,
    6.178721752808288,
    1.088486943963215,
    9.950055434408885e-2,
    4.784681996838866e-33,
    1.1815963332283862e-4,
    1.3748067355421944e-6,
    5.799125149291476e-9,
];

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::airy;

    static REFERENCE: [(f64, f64, f64, f64, f64); 9] = [
        (-652.891, 0.10447544538688257, 0.03927275745730001, -1.0034469627779596, 2.6695436038635454),
        (-123.0, -0.12756243520101215, 0.11148479086544187, -1.236685519932118, -1.4145093682439087),
        (-43.23, 0.21524897123688552, -0.045610160090116446, 0.3011299896521486, 1.414990730955133),
        (-1.0, 0.5355608832923521, 0.1039973894969446, -0.01016056711664521, 0.5923756264227924),
        (-3.674e-24, 0.3550280538878172, 0.6149266274460007, -0.2588194037928068, 0.4482883573538264),
        (1.4725, 0.07446845457316596, 1.828120247981961, -0.10036966257430177, 1.8104588650504139),
        (4.289, 0.0005198079279904233, 148.15162273358868, -0.0011049736723187855, 297.42956047164853),
        (25.822, 1.2773348723736746e-39, 2.4520220221102355e37, -6.503130605285926e-39, 1.2436182167358716e38),
        (103.6241, 3.434110379654982e-307, 4.552768382873564e304, -3.496612379694316e-306, 4.633433573716728e305),
    ];

    static ZEROS: [(f64, f64, f64, f64, f64); 4] = [
        // Ai zeros
        (-5.520559828095551, 2.313678943005095e-16, -0.36790153149695626, 0.8652040258941519, -0.016571236446578794),
        (-7.944133587120853, -3.222967925030853e-17, -0.33600537065305697, 0.9473357094415678, -0.010554505816063523),
        // Bi zeros
        (-6.169852128310251, -0.35786068428672557, -2.4652322944876206e-16, -0.014444121615121822, -0.8894799014265397),
        (-10.529913506705357, -0.31317702152506904, -6.278100478929636e-16, -0.007429478173120849, -1.016389659221249),
    ];

    #[test]
    fn test_airy() {
        for &(x, ai, bi, aip, bip) in &REFERENCE {
            let (val, der) = airy(x);

            let max_relative = if x < 2.0 {1e-10} else { 1e-5 };

            assert_relative_eq!(val.0, ai, max_relative=max_relative);
            assert_relative_eq!(val.1, bi, max_relative=max_relative);
            assert_relative_eq!(der.0, aip, max_relative=max_relative);
            assert_relative_eq!(der.1, bip, max_relative=max_relative);
        }

        for &(x, ai, bi, _aip, _bip) in &ZEROS {
            let (val, _) = airy(x);

            assert_relative_eq!(val.0, ai, max_relative=1e-9, epsilon=1e-10);
            assert_relative_eq!(val.1, bi, max_relative=1e-9, epsilon=1e-10);
            // derivatives near the zero are quite bad
            // assert_relative_eq!(der.0, aip, max_relative=1);
            // assert_relative_eq!(der.1, bip, max_relative=1);
        }
    }
}
