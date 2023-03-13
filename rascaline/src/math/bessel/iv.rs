use std::f64::consts::PI;

use log::warn;

use crate::math::gamma;


/// Modified Bessel function of the first kind, non-integer order
///
/// Returns modified Bessel function of order v of the argument. If x is
/// negative, v must be integer valued.
#[allow(clippy::float_cmp)]
pub fn bessel_iv(mut v: f64, x: f64) -> f64 {
    let mut sign;
    let mut t;
    let mut res = 0.0;

    if !(x.is_finite() && v.is_finite()) {
        return std::f64::NAN;
    }

    t = f64::floor(v);
    if v < 0.0 && t == v {
        v = -v;
        t = -t;
    }

    sign = 1;
    if x < 0.0 {
        if t != v {
            return std::f64::NAN;
        }
        if v != 2.0 * f64::floor(v / 2.0) {
            sign = -1;
        }
    }
    if x == 0.0 {
        if v == 0.0 {
            return 1.0;
        }
        if v < 0.0 {
            return std::f64::INFINITY;
        }
        return 0.0;
    }
    let ax = f64::abs(x);
    if f64::abs(v) > 50.0 {
        ikv_asymptotic_uniform(v, ax, &mut res, &mut 0.0);
    } else {
        res = ikv_temme(v, ax);
    }
    res *= sign as f64;
    return res;
}

fn iv_asymptotic(v: f64, x: f64) -> f64 {
    let mut sum;
    let mut term;
    let mut factor;
    let mut k;

    let prefactor = f64::exp(x) / f64::sqrt(2.0 * PI * x);
    if prefactor == std::f64::INFINITY {
        return prefactor;
    }
    let mu = 4.0 * v * v;
    sum = 1.0;
    term = 1.0;
    k = 1;
    loop {
        factor = (mu - ((2 * k - 1) * (2 * k - 1)) as f64) / (8.0 * x) / k as f64;
        if k > 100 {
            warn!("failed to converge bessel_iv function");
            break;
        }

        term *= -factor;
        sum += term;
        k += 1;
        if f64::EPSILON * f64::abs(sum) > f64::abs(term) {
            break;
        }
    }
    return sum * prefactor;
}

/// Uniform asymptotic expansion factors, (AMS5 9.3.9; AMS5 9.3.10)
static ASYMPTOTIC_UFACTORS: [[f64; 31]; 11] = [
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20833333333333334,
        0.0, 0.125, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3342013888888889, 0.0,
        -0.4010416666666667, 0.0, 0.0703125, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0258125964506173, 0.0, 1.8464626736111112,
        0.0, -0.8912109375, 0.0, 0.0732421875, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 4.669584423426247, 0.0, -11.207002616222995, 0.0,
        8.78912353515625, 0.0, -2.3640869140625, 0.0, 0.112152099609375, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -28.212072558200244, 0.0, 84.63621767460074, 0.0, -91.81824154324003,
        0.0, 42.53499874538846, 0.0, -7.368794359479631, 0.0, 0.22710800170898438,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 212.5701300392171,
        0.0, -765.2524681411816, 0.0, 1059.9904525279999, 0.0, -699.5796273761327,
        0.0, 218.1905117442116, 0.0, -26.491430486951554, 0.0, 0.5725014209747314,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1919.4576623184068, 0.0,
        8061.722181737308, 0.0, -13586.550006434136, 0.0, 11655.393336864536,
        0.0, -5305.646978613405, 0.0, 1200.9029132163525, 0.0, -108.09091978839464,
        0.0, 1.7277275025844574, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20204.29133096615, 0.0, -96980.5983886375,
        0.0, 192547.0012325315, 0.0, -203400.17728041555, 0.0, 122200.46498301747,
        0.0, -41192.65496889756, 0.0, 7109.514302489364, 0.0, -493.915304773088,
        0.0, 6.074042001273483, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, -242919.18790055133, 0.0, 1311763.614662977, 0.0, -2998015.918538106,
        0.0, 3763271.297656404, 0.0, -2813563.226586534, 0.0, 1268365.2733216248, 0.0,
        -331645.1724845636, 0.0, 45218.76898136274, 0.0, -2499.830481811209, 0.0,
        24.380529699556064, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        3284469.8530720375, 0.0, -19706819.11843222, 0.0, 50952602.49266463, 0.0,
        -74105148.21153264, 0.0, 66344512.27472903, 0.0, -37567176.66076335, 0.0,
        13288767.16642182, 0.0, -2785618.128086455, 0.0, 308186.40461266245, 0.0,
        -13886.089753717039, 0.0, 110.01714026924674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
];

/// Compute Iv, Kv from (AMS5 9.7.7 + 9.7.8), asymptotic expansion for large v
#[allow(clippy::many_single_char_names)]
fn ikv_asymptotic_uniform(mut v: f64, x: f64, i_value: &mut f64, k_value: &mut f64) {
    let mut term = 0.0;
    let mut divisor;
    let mut k;
    let mut n;
    let mut sign: i32 = 1;

    if v < 0.0 {
        sign = -1;
        v = -v;
    }

    let z = x / v;
    let t = 1.0 / f64::sqrt(1.0 + z * z);
    let t2 = t * t;
    let eta = f64::sqrt(1.0 + z * z) + f64::ln(z / (1.0 + 1.0 / t));
    let i_prefactor = f64::sqrt(t / (2.0 * PI * v)) * f64::exp(v * eta);
    let mut i_sum = 1.0;
    let k_prefactor = f64::sqrt(PI * t / (2.0 * v)) * f64::exp(-v * eta);
    let mut k_sum = 1.0;
    divisor = v;
    n = 1;
    while n < 11 {
        term = 0.0;
        k = 31 - 1 - 3 * n;
        while k < 31 - n {
            term *= t2;
            term += ASYMPTOTIC_UFACTORS[n as usize][k as usize];
            k += 2;
        }
        k = 1;
        while k < n {
            term *= t2;
            k += 2;
        }
        if n % 2 == 1 {
            term *= t;
        }
        term /= divisor;
        i_sum += term;
        k_sum += if n % 2 == 0 { term } else { -term };
        if f64::abs(term) < f64::EPSILON {
            break;
        }
        divisor *= v;
        n += 1;
    }
    if f64::abs(term) > 1e-3 * f64::abs(i_sum) {
        // sf_error(SF_ERROR_NO_RESULT);
        warn!("failed to converge bessel_iv function");
    }

    if f64::abs(term) > f64::EPSILON * f64::abs(i_sum) {
        // sf_error(SF_ERROR_LOSS);
        warn!("failed to converge bessel_iv function");
    }

    *k_value = k_prefactor * k_sum;

    if sign == 1 {
        *i_value = i_prefactor * i_sum;
    } else {
        *i_value = i_prefactor * i_sum
            + 2.0 / PI * f64::sin(PI * v) * k_prefactor * k_sum;
    }
}

/// Modified Bessel functions of the first and second kind of fractional order
///
/// Calculate K(v, x) and K(v+1, x) by method analogous to
/// Temme, Journal of Computational Physics, vol 21, 343 (1976)
#[allow(clippy::many_single_char_names)]
fn temme_ik_series(v: f64, x: f64, k: &mut f64, k1: &mut f64) -> i32 {
    let mut f;
    let mut h;
    let mut p;
    let mut q;
    let mut coef;
    let mut sum;
    let mut sum1;

    /*
     * |x| <= 2, Temme series converge rapidly
     * |x| > 2, the larger the |x|, the slower the convergence
     */

    let gp = gamma(v + 1.0) - 1.0;
    let gm = gamma(-v + 1.0) - 1.0;
    let a = f64::ln(x / 2.0);
    let b = f64::exp(v * a);
    let sigma = -a * v;
    let c = if f64::abs(v) < f64::EPSILON {
        1.0
    } else {
        f64::sin(PI * v) / (v * PI)
    };

    let d = if f64::abs(sigma) < f64::EPSILON {
        1.0
    } else {
        f64::sinh(sigma) / sigma
    };

    let gamma1 = if f64::abs(v) < f64::EPSILON {
        -0.5772156649015329
    } else {
        0.5f32 as f64 / v * (gp - gm) * c
    };

    let gamma2 = (2.0 + gp + gm) * c / 2.0;

    /* initial values */
    p = (gp + 1.0) / (2.0 * b);
    q = (1.0 + gm) * b / 2.0;
    f = (f64::cosh(sigma) * gamma1 + d * -a * gamma2) / c;
    h = p;
    coef = 1.0;
    sum = coef * f;
    sum1 = coef * h;

    let mut ik = 1.0;
    while ik < 500.0 {
        f = (ik * f + p + q) / (ik * ik - v * v);
        p /= ik - v;
        q /= ik + v;
        h = p - ik * f;
        coef *= x * x / (4.0 * ik);
        sum += coef * f;
        sum1 += coef * h;
        if f64::abs(coef * f) < f64::abs(sum) * f64::EPSILON {
            break;
        }
        ik += 1.0;
    }

    if ik >= 500.0 {
        warn!("failed to converge bessel_iv function");
    }

    *k = sum;
    *k1 = 2.0 * sum1 / x;
    return 0;
}

/// Evaluate continued fraction `fv = I_(v+1) / I_v`, derived from
/// Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73
#[allow(clippy::many_single_char_names)]
fn cf1_ik(v: f64, x: f64, fv: &mut f64) -> i32 {
    let mut c;
    let mut d;
    let mut f;
    let mut a;
    let mut b;
    let mut delta;

    /*
     * |x| <= |v|, CF1_ik converges rapidly
     * |x| > |v|, CF1_ik needs O(|x|) iterations to converge
     */

    /*
     * modified Lentz's method, see
     * Lentz, Applied Optics, vol 15, 668 (1976)
     */

    let tiny = 1.0 / f64::sqrt(1.7976931348623157e308);

    f = tiny;
    c = f;
    d = 0.0;

    let mut k = 1;
    while k < 500 {
        a = 1.0;
        b = 2.0 * (v + k as f64) / x;
        c = b + a / c;
        d = b + a * d;
        if c == 0.0 {
            c = tiny;
        }
        if d == 0.0 {
            d = tiny;
        }
        d = 1.0 / d;
        delta = c * d;
        f *= delta;
        if f64::abs(delta - 1.0) <= 2.0 * f64::EPSILON {
            break;
        }
        k += 1;
    }

    if k == 500 {
        warn!("failed to converge bessel_iv function");
    }

    *fv = f;
    return 0;
}


/// Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
/// `z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x)`, see
/// Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
#[allow(non_snake_case, clippy::many_single_char_names)]
fn cf2_ik(v: f64, x: f64, Kv: &mut f64, Kv1: &mut f64) -> i32 {
    let mut S;
    let mut C;
    let mut Q;
    let mut D;
    let mut f;
    let mut a;
    let mut b;
    let mut q;
    let mut delta;

    let mut current;
    let mut prev;

    /*
     * |x| >= |v|, CF2_ik converges rapidly
     * |x| -> 0, CF2_ik fails to converge
     */

    /*
     * Steed's algorithm, see Thompson and Barnett,
     * Journal of Computational Physics, vol 64, 490 (1986)
     */

    a = v * v - 0.25f32 as f64;
    b = 2.0 * (x + 1.0);
    D = 1.0 / b;
    delta = D;
    f = delta;
    prev = 0.0;
    current = 1.0;
    C = -a;
    Q = C;
    S = 1.0 + Q * delta;

    let mut ik = 2_u64;
    while ik < 500 {
        a -= (2 * (ik - 1)) as f64;
        b += 2.0;
        D = 1.0 / (b + a * D);
        delta *= b * D - 1.0;
        f += delta;
        q = (prev - (b - 2.0) * current) / a;
        prev = current;
        current = q;
        C *= -a / ik as f64;
        Q += C * q;
        S += Q * delta;
        if f64::abs(Q * delta) < f64::abs(S) * f64::EPSILON {
            break;
        }
        ik += 1;
    }
    if ik == 500 {
        warn!("failed to converge bessel_iv function");
    }
    *Kv = f64::sqrt(PI / (2.0 * x)) * f64::exp(-x) / S;
    *Kv1 = *Kv * (0.5f32 as f64 + v + x + (v * v - 0.25f32 as f64) * f) / x;
    return 0;
}


/// Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
/// Temme, Journal of Computational Physics, vol 19, 324 (1975)
#[allow(non_snake_case, clippy::many_single_char_names)]
fn ikv_temme(mut v: f64, x: f64) -> f64 {
    // modification from scipy's cephes: this only returns Iv, never Kv

    let mut Iv;
    let mut Ku = 0.0;
    let mut Ku1 = 0.0;
    let mut fv = 0.0;
    let mut current;
    let mut prev;
    let mut next;
    let mut reflect = false;

    if v < 0.0 {
        v = -v;
        reflect = true;
    }

    let n = f64::round(v) as u32;
    let u = v - n as f64;
    if x < 0.0 {
        return std::f64::NAN;
    }

    if x == 0.0 {
        Iv = if v == 0.0 { 1.0 } else { 0.0 };

        if reflect {
            let z = u + n.wrapping_rem(2) as f64;
            Iv = if f64::sin(PI * z) == 0.0 {
                Iv
            } else {
                std::f64::INFINITY
            };
        }

        return Iv;
    }

    /* x is positive until reflection */
    let W = 1.0 / x; /* Wronskian */
    if x <= 2.0 {
        temme_ik_series(u, x, &mut Ku, &mut Ku1); /* Temme series */
    } else {
        /* x in (2, \infty) */
        cf2_ik(u, x, &mut Ku, &mut Ku1); /* continued fraction CF2_ik */
    }
    prev = Ku;
    current = Ku1;

    for ik in 1..=n { /* forward recurrence for K */
        next = 2.0 * (u + ik as f64) * current / x + prev;
        prev = current;
        current = next;
    }
    let Kv = prev;
    let Kv1 = current;

    let mut lim: f64 = (4.0 * v * v + 10.0) / (8.0 * x);
    lim *= lim;
    lim *= lim;
    lim /= 24.0;
    if lim < f64::EPSILON * 10.0 && x > 100.0 {
        /*
	     * x is huge compared to v, CF1 may be very slow
	     * to converge so use asymptotic expansion for large
	     * x case instead.  Note that the asymptotic expansion
	     * isn't very accurate - so it's deliberately very hard
	     * to get here - probably we're going to overflow:
	     */
        Iv = iv_asymptotic(v, x);
    } else {
        cf1_ik(v, x, &mut fv); /* continued fraction CF1_ik */
        Iv = W / (Kv * fv + Kv1); /* Wronskian relation */
    }

    if reflect {
        let z_0 = u + n.wrapping_rem(2) as f64;
        return Iv + 2.0 / PI * f64::sin(PI * z_0) * Kv; /* reflection formula */
    }

    return Iv;
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    // computed using mpmath, converting overflow (>1e308) to inf and underflow
    // (<1e-323) to 0.0
    static REFERENCE: [(f64, f64, f64); 92] = [
        (-100.3, 1.0, 1.48788821588229e+186),
        (-100.3, 10.0, 5.81385536644804e+85),
        (-100.3, 200.5, 6.58080030590051e+74),
        (-100.3, 401.0, 1.05963257281431e+167),
        (-100.3, 600.5, 2.36123204175185e+255),
        (-100.3, 700.6, 2.13787442407607e+299),
        (-100.3, 1300.0, std::f64::INFINITY),
        (-20.0, 1.0, 3.96683598581902e-25),
        (-20.0, 10.0, 0.000125079973564495),
        (-20.0, 200.5, 1.23661306184996e+85),
        (-20.0, 401.0, 1.71683807030892e+172),
        (-20.0, 600.5, 7.25819963948107e+258),
        (-20.0, 700.6, 2.09367143600173e+302),
        (-20.0, 1300.0, std::f64::INFINITY),
        (-10.0, 1.0, 2.75294803983687e-10),
        (-10.0, 10.0, 21.8917061637234),
        (-10.0, 200.5, 2.6158708155538e+85),
        (-10.0, 401.0, 2.49657353350645e+172),
        (-10.0, 600.5, 9.31944446151452e+258),
        (-10.0, 700.6, 2.59388351323027e+302),
        (-10.0, 1300.0, std::f64::INFINITY),
        (-1.0, 1.0, 0.565159103992485),
        (-1.0, 10.0, 2670.98830370125),
        (-1.0, 200.5, 3.35028839470096e+85),
        (-1.0, 401.0, 2.82500017764435e+172),
        (-1.0, 600.5, 1.0120885431321e+259),
        (-1.0, 700.6, 2.78391772501927e+302),
        (-1.0, 1300.0, std::f64::INFINITY),
        (-0.5, 1.0, 1.23120021459297),
        (-0.5, 10.0, 2778.78461532957),
        (-0.5, 200.5, 3.35657610792771e+85),
        (-0.5, 401.0, 2.82764655074214e+172),
        (-0.5, 600.5, 1.01272129651641e+259),
        (-0.5, 700.6, 2.78540929649693e+302),
        (-0.5, 1300.0, std::f64::INFINITY),
        (0.0, -1300.0, std::f64::INFINITY),
        (0.0, -11.0, 7288.48933982125),
        (0.0, -10.0, 2815.71662846625),
        (0.0, -1.0, 1.26606587775201),
        (0.0, 1.0, 1.26606587775201),
        (0.0, 10.0, 2815.71662846625),
        (0.0, 200.5, 3.35867463800083e+85),
        (0.0, 401.0, 2.82852922635177e+172),
        (0.0, 600.5, 1.01293230225787e+259),
        (0.0, 700.6, 2.7859066646434e+302),
        (0.0, 1300.0, std::f64::INFINITY),
        (1.0, -1300.0, -std::f64::INFINITY),
        (1.0, -11.0, -6948.85865981216),
        (1.0, -10.0, -2670.98830370125),
        (1.0, -1.0, -0.565159103992485),
        (1.0, 1.0, 0.565159103992485),
        (1.0, 10.0, 2670.98830370125),
        (1.0, 200.5, 3.35028839470096e+85),
        (1.0, 401.0, 2.82500017764435e+172),
        (1.0, 600.5, 1.0120885431321e+259),
        (1.0, 700.6, 2.78391772501927e+302),
        (1.0, 1300.0, std::f64::INFINITY),
        (12.49, 1.0, 1.06214483463358e-13),
        (12.49, 10.0, 1.85487470997247),
        (12.49, 200.5, 2.27429677988355e+85),
        (12.49, 401.0, 2.3280143028664e+172),
        (12.49, 600.5, 8.89455243908991e+258),
        (12.49, 700.6, 2.49219423462741e+302),
        (12.49, 1300.0, std::f64::INFINITY),
        (120.0, -1300.0, std::f64::INFINITY),
        (120.0, -11.0, 1.33841738592428e-110),
        (120.0, -10.0, 1.38248722487846e-115),
        (120.0, -1.0, 1.12694823613939e-235),
        (120.0, 1.0, 1.12694823613939e-235),
        (120.0, 10.0, 1.38248722487846e-115),
        (120.0, 200.5, 2.08641377637118e+70),
        (120.0, 401.0, 5.02466934582507e+164),
        (120.0, 600.5, 6.47517615090698e+253),
        (120.0, 700.6, 9.75968494448862e+297),
        (120.0, 1300.0, std::f64::INFINITY),
        (301.0, -1300.0, -std::f64::INFINITY),
        (301.0, -11.0, 0.0),
        (301.0, -10.0, 0.0),
        (301.0, -1.0, 0.0),
        (301.0, 1.0, 0.0),
        (301.0, 10.0, 0.0),
        (301.0, 200.5, 0.131118954561987),
        (301.0, 401.0, 2.14246804623493e+125),
        (301.0, 600.5, 7.21577161334608e+226),
        (301.0, 700.6, 5.68946923641546e+274),
        (301.0, 1300.0, std::f64::INFINITY),
        (-59.5, 3.5, -1.88381631760945e+64),
        (-39.5, 3.5, -2.40435409707822e+35),
        (-19.5, 3.5, -136430465086.747),
        (0.5, 3.5, 7.0552194086912),
        (20.5, 3.5, 9.98384982441339e-15),
        (40.5, 3.5, 1.43995556110118e-39),
    ];

    #[test]
    fn test_bessel_iv() {
        for (v, z, expected) in REFERENCE {
            assert_relative_eq!(super::bessel_iv(v, z), expected, max_relative=1e-12);
        }
    }
}
