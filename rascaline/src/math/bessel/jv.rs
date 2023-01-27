#![allow(clippy::many_single_char_names, clippy::similar_names, clippy::float_cmp)]
#![allow(clippy::manual_range_contains, clippy::too_many_lines)]

use gamma::ln_gamma;

use crate::math::gamma;

use super::{eval_polynomial, bessel_j0, bessel_j1, bessel_y0, bessel_y1};
use super::{MAXLOG};

const CBRT_TWO: f64 = 1.2599210498948732;


/// Spherical Bessel function of the first kind of non-integer order
pub fn spherical_bessel_jv(n: f64, x: f64) -> f64 {
    return f64::sqrt(std::f64::consts::PI / (2.0 * x)) * bessel_jv(n + 0.5, x);
}

/// Spherical Bessel function of the second kind of non-integer order
pub fn spherical_bessel_yv(n: f64, x: f64) -> f64 {
    return f64::sqrt(std::f64::consts::PI / (2.0 * x)) * bessel_yv(n + 0.5, x);
}

/// Bessel function of second kind of integer order
///
/// Returns Bessel function of order n, where n is a (possibly negative)
/// integer.
///
/// The function is evaluated by forward recurrence on n, starting with values
/// computed by the routines y0() and y1().
///
/// If n = 0 or 1 the routine for y0 or y1 is called directly.
///
/// # Accuracy
///
/// ```text
///                      Absolute error, except relative
///                      when y > 1:
/// arithmetic   domain     # trials      peak         rms
///    IEEE      0, 30       30000       3.4e-15     4.3e-16
/// ````
///
/// Spot checked against tables for x, n between 0 and 100.
pub fn bessel_yn(mut n: i32, x: f64) -> f64 {
    let sign;
    if n < 0 {
        n = -n;
        if (n & 1) == 0 {	/* -1**n */
            sign = 1.0;
        } else {
            sign = -1.0;
        }
    } else {
        sign = 1.0;
    }

    if n == 0 {
	    return sign * bessel_y0(x);
    } else if n == 1 {
	    return sign * bessel_y1(x);
    }

    // test for overflow
    if x == 0.0 {
        // sf_error(SF_ERROR_SINGULAR);
        return -f64::INFINITY * sign;
    } else if x < 0.0 {
        // sf_error(SF_ERROR_DOMAIN);
        return f64::NAN;
    }

    // forward recurrence on n
    let mut anm2 = bessel_y0(x);
    let mut anm1 = bessel_y1(x);
    let mut r = 2.0;
    let mut an = 0.0;
    for _ in 1..n {
        an = r * anm1 / x - anm2;
        anm2 = anm1;
        anm1 = an;
        r += 2.0;
    }

    return sign * an;
}

// Bessel function of second kind, non-integer order
pub fn bessel_yv(v: f64, x: f64) -> f64 {
    let n = v as i32;
    if n as f64 == v {
        return bessel_yn(n, x);
    } else if v == f64::floor(v) {
        // sf_error(SF_ERROR_DOMAIN);
        return std::f64::NAN;
    }

    let t = std::f64::consts::PI * v;
    let y = (f64::cos(t) * bessel_jv(v, x) - bessel_jv(-v, x)) / f64::sin(t);
    if y.is_infinite() {
        if v > 0.0 {
            // sf_error(SF_ERROR_OVERFLOW);
            return -std::f64::INFINITY;
        } else if v < -1e10 {
            // sf_error(SF_ERROR_DOMAIN);
            return std::f64::NAN;
        }
    }

    return y;
}



/// Bessel function of the first kind, non-integer order
///
/// Returns Bessel function of order v of the argument, where v is real.
/// Negative x is allowed if v is an integer.
///
/// Several expansions are included: the ascending power series, the Hankel
/// expansion, and two transitional expansions for large v.  If v is not too
/// large, it is reduced by recurrence to a region of best accuracy. The
/// transitional expansions give 12D accuracy for v > 500.
///
/// # Accuracy
///
/// Results for integer v are indicated by *, where x and v both vary from -125
/// to +125.  Otherwise, x ranges from 0 to 125, v ranges as indicated by
/// "domain." Error criterion is absolute, except relative when |jv()| > 1.
///
/// ```text
/// arithmetic  v domain  x domain    # trials      peak       rms
///    IEEE      0,125     0,125      100000      4.6e-15    2.2e-16
///    IEEE   -125,0       0,125       40000      5.4e-11    3.7e-13
///    IEEE      0,500     0,500       20000      4.4e-15    4.0e-16
/// Integer v:
///    IEEE   -125,125   -125,125      50000      3.5e-15*   1.9e-16*
/// ```
pub fn bessel_jv(mut n: f64, mut x: f64) -> f64 {
    let mut k;
    let mut q;
    let mut t;
    let i;

    let mut sign = 1.0;
    let mut nint = 0;
    let an = f64::abs(n);
    let mut y = f64::floor(an);

    if an == y {
        nint = 1;
        i = (an - 16384.0 * f64::floor(an / 16384.0)) as i32;
        if n < 0.0 {
            if i & 1 != 0 {
                sign = -sign;
            }
            n = an;
        }

        if x < 0.0 {
            if i & 1 != 0 {
                sign = -sign;
            }
            x = -x;
        }

        if n == 0.0 {
            return bessel_j0(x);
        }

        if n == 1.0 {
            return sign * bessel_j1(x);
        }
    }

    if x < 0.0 && y != an {
        // sf_error(SF_ERROR_DOMAIN);
        return std::f64::NAN;
    }

    if x == 0.0 && n < 0.0 && nint == 0 {
        // sf_error(SF_ERROR_OVERFLOW);
        return std::f64::INFINITY / gamma(n + 1.0);
    }

    y = f64::abs(x);
    if y * y < f64::abs(n + 1.0) * f64::EPSILON {
        return f64::powf(0.5 * x, n) / gamma(n + 1.0);
    }

    k = 3.6 * f64::sqrt(y);
    t = 3.6 * f64::sqrt(an);

    if y < t && an > 21.0 {
        return sign * jvs(n, x);
    }

    if an < k && y > 21.0 {
        return sign * hankel(n, x);
    }

    if an < 500.0 {
    	// Note: if x is too large, the continued fraction will fail; but
    	// then the Hankel expansion can be used.
    	if nint != 0 {
    	    k = 0.0;
    	    q = recur(&mut n, x, &mut k, 1);

    	    if k == 0.0 {
    		    y = bessel_j0(x) / q;
    		    return sign * y;
    	    }

            if k == 1.0 {
    		    y = bessel_j1(x) / q;
    		    return sign * y;
    	    }
    	}

        if (an > 2.0 * y) || (n >= 0.0) && (n < 20.0) && (y > 6.0) && (y < 20.0) {
            // Recurse backward from a larger value of n
            k = n;

            y = y + an + 1.0;
            if y < 30.0 {
                y = 30.0;
            }

            y = n + f64::floor(y - n);
            q = recur(&mut y, x, &mut k, 0);
            y = jvs(y, x) * q;

            return sign * y;
        }

        if k <= 30.0 {
            k = 2.0;
        } else if k < 90.0 {
            k = 3.0 * k / 4.0;
        }

        if an > (k + 3.0) {
            if n < 0.0 {
                k = -k;
            }

            q = n - f64::floor(n);
            k = f64::floor(k) + q;
            if n > 0.0 {
                q = recur(&mut n, x, &mut k, 1);
            } else {
                t = k;
                k = n;
                q = recur(&mut t, x, &mut k, 1);
                k = t;
            }

            if q == 0.0 {
                y = 0.0;
                return sign * y;
            }
        } else {
            k = n;
            q = 1.0;
        }

        // boundary between convergence of power series and Hankel expansion
        y = f64::abs(k);
        if y < 26.0 {
            t = (0.0083 * y + 0.09) * y + 12.9;
        } else {
            t = 0.9 * y;
        }

        if x > t {
            y = hankel(k, x);
        } else {
            y = jvs(k, x);
        }

        if n > 0.0 {
            y /= q;
        } else {
            y *= q;
        }
    } else {
        // For large n, use the uniform expansion or the transitional expansion.
        // But if x is of the order of n**2, these may blow up, whereas the
        // Hankel expansion will then work.
        if n < 0.0 {
            // sf_error(SF_ERROR_LOSS);
            y = std::f64::NAN;
        } else {
            t = x / n;
            t /= n;
            if t > 0.3 {
                y = hankel(n, x);
            } else {
                y = jnx(n, x);
            }
        }
    }

    return sign * y;
}

/// Reduce the order by backward recurrence.
/// AMS55 #9.1.27 and 9.1.73.
fn recur(n: &mut f64, x: f64, new_n: &mut f64, cancel: i32) -> f64 {
    const BIG: f64 = 1.4411518807585587e17;

    let mut pkm2;
    let mut pkm1;
    let mut pk;
    let mut qkm2;
    let mut qkm1;
    let mut k;
    let mut ans;
    let mut qk;
    let mut xk;
    let mut yk;
    let mut r;
    let mut t;
    let mut ctr;

    let maxiter = 22000;
    let mut miniter = (f64::abs(x) - f64::abs(*n)) as i32;
    if miniter < 1 {
        miniter = 1;
    }

    let mut nflag;
    if *n < 0.0 {
        nflag = 1;
    } else {
        nflag = 0;
    }

    loop {
        pkm2 = 0.0;
        qkm2 = 1.0;
        pkm1 = x;
        qkm1 = *n + *n;
        xk = -x * x;
        yk = qkm1;
        ans = 0.0;
        ctr = 0;

        loop {
            yk += 2.0;
            pk = pkm1 * yk + pkm2 * xk;
            qk = qkm1 * yk + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if qk != 0.0 && ctr > miniter {
                r = pk / qk;
            } else {
                r = 0.0;
            }

            if r == 0.0 {
                t = 1.0;
            } else {
                t = f64::abs((ans - r) / r);
                ans = r;
            }

            ctr += 1;
            if ctr > maxiter {
                // sf_error(SF_ERROR_UNDERFLOW);
                break;
            }

            if t < f64::EPSILON {
                break;
            }

            if f64::abs(pk) > BIG {
                pkm2 /= BIG;
                pkm1 /= BIG;
                qkm2 /= BIG;
                qkm1 /= BIG;
            }

            if t < f64::EPSILON {
                break;
            }
        }
        if ans == 0.0 {
            ans = 1.0;
        }
        if nflag <= 0 {
            break;
        }
        if f64::abs(ans) >= 0.125 {
            break;
        }
        nflag = -(1);
        *n -= 1.0;
    }

    let kf = *new_n;
    pk = 1.0;
    pkm1 = 1.0 / ans;
    k = *n - 1.0;
    r = 2.0 * k;
    loop {
        pkm2 = (pkm1 * r - pk * x) / x;
        pk = pkm1;
        pkm1 = pkm2;
        r -= 2.0;
        k -= 1.0;

        if k <= kf + 0.5 {
            break;
        }
    }
    if cancel != 0 && kf >= 0.0 && f64::abs(pk) > f64::abs(pkm1) {
        k += 1.0;
        pkm2 = pk;
    }
    *new_n = k;
    return pkm2;
}

// https://stackoverflow.com/a/55696477/4692076
fn float_exponent(s: f64) -> f64 {
    if s == 0.0 {
        return 0.0;
    }

    let lg = s.abs().log2();
    let exp = lg.floor() + 1.0;
    return exp;
}

/// Ascending power series for Jv(x).
/// AMS55 #9.1.10.
fn jvs(n: f64, x: f64) -> f64 {
    let z = -x * x / 4.0;
    let mut u = 1.0;
    let mut y = u;
    let mut k = 1.0;
    let mut t = 1.0;
    while t > f64::EPSILON {
        u *= z / (k * (n + k));
        y += u;
        k += 1.0;
        if y != 0.0 {
            t = f64::abs(u / y);
        }
    }
    let mut ex = float_exponent(0.5 * x);
    ex *= n;

    if ex > -1023.0 && ex < 1023.0 && n > 0.0 && n < 171.6243769563027 - 1.0 {
        t = f64::powf(0.5 * x, n) / gamma(n + 1.0);
        y *= t;
    } else {
        dbg!("here");
        t = n * f64::ln(0.5 * x) - ln_gamma(n + 1.0);
        if y < 0.0 {
            y = -y;
        }
        t += f64::ln(y);
        if t < -MAXLOG {
            return 0.0;
        }
        if t > MAXLOG {
            // sf_error(SF_ERROR_OVERFLOW);
            return std::f64::INFINITY;
        }
        y = f64::exp(t);
    }
    return y;
}


// Hankel's asymptotic expansion for large x.
// AMS55 #9.2.5.
fn hankel(n: f64, x: f64) -> f64 {
    let m = 4.0 * n * n;
    let mut j = 1.0;
    let z = 8.0 * x;
    let mut k = 1.0;
    let mut p = 1.0;
    let mut u = (m - 1.0) / z;
    let mut q = u;
    let mut sign = 1.0;
    let mut conv = 1.0;
    let mut flag = 0;
    let mut t = 1.0;
    let mut pp = 1.0e38;
    let mut qq = 1.0e38;
    while t > f64::EPSILON {
        k += 2.0;
        j += 1.0;
        sign = -sign;
        u *= (m - k * k) / (j * z);
        p += sign * u;
        k += 2.0;
        j += 1.0;
        u *= (m - k * k) / (j * z);
        q += sign * u;
        t = f64::abs(u / p);
        if t < conv {
            conv = t;
            qq = q;
            pp = p;
            flag = 1;
        }
        if flag != 0 && t > conv {
            break;
        }
    }
    u = x - (0.5 * n + 0.25) * std::f64::consts::PI;
    t = f64::sqrt(2.0 / (std::f64::consts::PI * x)) * (pp * f64::cos(u) - qq * f64::sin(u));
    return t;
}

static LAMBDA: [f64; 11] = [
    1.0,
    1.0416666666666667e-1,
    8.355034722222222e-2,
    1.2822657455632716e-1,
    2.9184902646414046e-1,
    8.816272674437576e-1,
    3.3214082818627677,
    1.4995762986862555e1,
    7.892301301158652e1,
    4.744515388682643e2,
    3.207490090890662e3,
];

static MU: [f64; 11] = [
    1.0,
    -1.4583333333333334e-1,
    -9.874131944444445e-2,
    -1.4331205391589505e-1,
    -3.1722720267841353e-1,
    -9.424291479571203e-1,
    -3.5112030408263544,
    -1.5727263620368046e1,
    -8.228143909718594e1,
    -4.923553705236705e2,
    -3.3162185685479726e3,
];

static P1: [f64; 2] = [
    -2.0833333333333334e-1,
    1.25e-1
];

static P2: [f64; 3] = [
    3.342013888888889e-1,
    -4.010416666666667e-1,
    7.03125e-2,
];

static P3: [f64; 4] = [
    -1.0258125964506173,
    1.8464626736111112,
    -8.912109375e-1,
    7.32421875e-2,
];

static P4: [f64; 5] = [
    4.669584423426247,
    -1.1207002616222994E1,
    8.78912353515625,
    -2.3640869140625,
    1.12152099609375e-1,
];

static P5: [f64; 6] = [
    -2.8212072558200244e1,
    8.463621767460073e1,
    -9.181824154324002e1,
    4.253499874538846e1,
    -7.368794359479632,
    2.2710800170898438e-1,
];

static P6: [f64; 7] = [
    2.1257013003921713e2,
    -7.652524681411817e2,
    1.0599904525279999e3,
    -6.995796273761325e2,
    2.181905117442116e2,
    -2.6491430486951554e1,
    5.725014209747314e-1,
];

static P7: [f64; 8] = [
    -1.919457662318407e3,
    8.061722181737309e3,
    -1.3586550006434138e4,
    1.1655393336864534e4,
    -5.305646978613403e3,
    1.2009029132163525e3,
    -1.0809091978839466e2,
    1.7277275025844574,
];

/// Asymptotic expansion for large n.
/// AMS55 #9.3.35.
#[allow(clippy::too_many_lines)]
fn jnx(n: f64, x: f64) -> f64 {
    let zeta;
    let mut t;
    let sz;
    let mut sign;
    let nflg;
    let mut u = [0.0; 8];

    let cbn = f64::cbrt(n);
    let z = (x - n) / cbn;
    if f64::abs(z) <= 0.7 {
        return jnt(n, x);
    }
    let z = x / n;
    let zz = 1.0 - z * z;

    if zz == 0.0 {
        return 0.0;
    }

    if zz > 0.0 {
        sz = f64::sqrt(zz);
        t = 1.5 * (f64::ln((1.0 + sz) / z) - sz);
        zeta = f64::cbrt(t * t);
        nflg = 1;
    } else {
        sz = f64::sqrt(-zz);
        t = 1.5 * (sz - f64::acos(1.0 / z));
        zeta = -f64::cbrt(t * t);
        nflg = -(1);
    }

    let z32i = f64::abs(1.0 / t);
    let sqz = f64::cbrt(t);
    let n23 = f64::cbrt(n * n);
    t = n23 * zeta;
    let ((ai, _), (aip, _)) = super::airy(t);

    u[0] = 1.0;

    let zzi = 1.0 / zz;
    u[1] = eval_polynomial(zzi, &P1) / sz;
    u[2] = eval_polynomial(zzi, &P2) / zz;
    u[3] = eval_polynomial(zzi, &P3) / (sz * zz);

    let mut pp = zz * zz;
    u[4] = eval_polynomial(zzi, &P4) / pp;
    u[5] = eval_polynomial(zzi, &P5) / (pp * sz);

    pp *= zz;
    u[6] = eval_polynomial(zzi, &P6) / pp;
    u[7] = eval_polynomial(zzi, &P7) / (pp * sz);

    pp = 0.0;
    let mut qq = 0.0;

    let mut np = 1.0;
    let mut do_a = true;
    let mut do_b = true;
    let mut akl = std::f64::INFINITY;
    let mut bkl = std::f64::INFINITY;
    let mut k = 0;
    while k <= 3 {
        let tk = 2 * k;
        let tkp1 = tk + 1;
        let mut zp = 1.0;
        let mut ak = 0.0;
        let mut bk = 0.0;
        let mut s = 0;
        while s <= tk {
            if do_a {
                if s & 3 > 1 {
                    sign = nflg as f64;
                } else {
                    sign = 1.0;
                }
                ak += sign * MU[s] * zp * u[(tk - s)];
            }

            if do_b {
                let m = tkp1 - s;
                if (m + 1) & 3 > 1 {
                    sign = nflg as f64;
                } else {
                    sign = 1.0;
                }
                bk += sign * LAMBDA[s] * zp * u[m];
            }
            zp *= z32i;
            s += 1;
        }

        if do_a {
            ak *= np;
            t = f64::abs(ak);
            if t < akl {
                akl = t;
                pp += ak;
            } else {
                do_a = false;
            }
        }

        if do_b {
            bk += LAMBDA[tkp1] * zp * u[0];
            bk *= -np / sqz;
            t = f64::abs(bk);
            if t < bkl {
                bkl = t;
                qq += bk;
            } else {
                do_b = false;
            }
        }
        if np < f64::EPSILON {
            break;
        }
        np /= n * n;
        k += 1;
    }

    t = 4.0 * zeta / zz;
    t = f64::sqrt(f64::sqrt(t));
    t *= ai * pp / f64::cbrt(n) + aip * qq / (n23 * n);
    return t;
}

static PF2: [f64; 2] = [
    -9e-2,
    8.571428571428572e-2,
];

static PF3: [f64; 3] = [
    1.367142857142857e-1,
    -5.492063492063492e-2,
    -4.4444444444444444e-3,
];

static PF4: [f64; 4] = [
    1.35e-3,
    -1.6036054421768708e-1,
    4.259018759018759e-2,
    2.733044733044733e-3,
];

static PG1: [f64; 2] = [
    -2.4285714285714285e-1,
    1.4285714285714285e-2,
];

static PG2: [f64; 3] = [
    -9e-3,
    1.9396825396825396e-1,
    -1.1746031746031746e-2,
];

static PG3: [f64; 3] = [
    1.9607142857142858e-2,
    -1.5983694083694083e-1,
    6.383838383838384e-3,
];


/// Asymptotic expansion for transition region, n large and x close to n.
/// AMS55 #9.3.23.
fn jnt(n: f64, x: f64) -> f64 {
    let cbn = f64::cbrt(n);
    let z = (x - n) / cbn;

    let ((ai, _), (aip, _)) = super::airy(-CBRT_TWO * z);

    let zz = z * z;
    let z3 = zz * z;

    let f = [
        1.0,
        -z / 5.0,
        eval_polynomial(z3, &PF2) * zz,
        eval_polynomial(z3, &PF3),
        eval_polynomial(z3, &PF4) * z,
    ];


    let g = [
        0.3 * zz,
        eval_polynomial(z3, &PG1),
        eval_polynomial(z3, &PG2) * z,
        eval_polynomial(z3, &PG3) * zz,
    ];

    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut nk = 1.0;
    let n23 = f64::cbrt(n * n);

    for k in 0..5 {
        let fk = f[k] * nk;
        pp += fk;
        if k != 4 {
            qq += g[k] * nk;
        }
        nk /= n23;
    }

    return CBRT_TWO * ai * pp / cbn + f64::cbrt(4.0) * aip * qq / n;
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_jv() {
        todo!()
    }

    #[test]
    fn test_yv() {
        todo!()
    }

    #[test]
    fn test_large_n() {
        assert_relative_eq!(1e80 * bessel_jv(171.6, 45.4), 8.938521429926956, max_relative=1e-12);
        assert_relative_eq!(1e80 * bessel_jv(176.0, -45.4), 0.0012279789648073855, max_relative=1e-12);
    }
}
