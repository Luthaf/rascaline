//! Calaculates spherical Bessel functions
//! Translated from scipy (specfun.f)

use std::f64;

/// Computes the the spherical Bessel functions of the first kind j_n(x) 
/// and their derivatives. n is the maximum order, x is the argument. The 
/// function returns two vectors: the first contains the values of the 
/// spherical Bessel functions up to and including order n, while the 
/// second contains the corresponding derivatives with respect to x.
pub fn spherical_bessel_first_kind(n: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    
    if x > 1500.0 {
        panic!("The spherical Bessel implementation 
        does not support large arguments (>1500)");
    }
    if x.is_nan() {
        panic!("NaN was fed to spherical Bessel function");
    }
    
    let mut sj = vec![0.0; n+1];
    let mut dj = vec![0.0; n+1];
    let n = n as i32;
    let mut nm = n;
    if x.abs() < 1.0e-100 {
        sj[0] = 1.0;
        if n > 0 {
            dj[1] = 1.0/3.0;
        }
        return (sj, dj);
    }
    sj[0] = f64::sin(x)/x;
    dj[0] = (f64::cos(x)-f64::sin(x)/x)/x;
    if n < 1 {
        return (sj, dj);
    }
    sj[1] = (sj[0]-f64::cos(x))/x;
    if n >= 2 {
        let sa = sj[0];
        let sb = sj[1];
        let mut m = msta1(x, 200);
        if m < n {
            nm = m; 
        }
        else
        {
            m = msta2(x, n, 15);
        }
        let mut f = 0.0;
        let mut f0 = 0.0;
        let mut f1 = 1.0 - 100.0;

        for k in (0..m+1).rev().map(|k| k as usize) {
            f = (2.0*(k as f64)+3.0)*f1/x-f0;
            if (k as i32) <= nm {
                sj[k] = f;
            } 
            f0 = f1;
            f1 = f;
        }
        let mut cs = 0.0;
        if sa.abs() > sb.abs() { 
            cs = sa/f;
        }
        if sa.abs() <= sb.abs() { 
            cs = sb/f0;
        }
        for k in (0..nm+1).map(|k| k as usize) {
            sj[k]=cs*sj[k];
        }
    }
    for k in (1..nm+1).map(|k| k as usize) {
        dj[k] = sj[k-1]-((k as f64)+1.0)*sj[k]/x;
    }
    return (sj, dj);
}

/// Computes the the spherical Bessel functions of the second kind y_n(x) 
/// and their derivatives. n is the maximum order, x is the argument. The 
/// function returns two vectors: the first contains the values of the 
/// spherical Bessel functions up to and including order n, while the 
/// second contains the corresponding derivatives with respect to x.
pub fn spherical_bessel_second_kind(n: usize, x: f64) -> (Vec<f64>, Vec<f64>) {

    if x > 1500.0 {
        panic!("The spherical Bessel implementation 
        does not support large arguments (>1500)");
    }
    if x.is_nan() {
        panic!("NaN was fed to spherical Bessel function");
    }

    let mut sy = vec![0.0; n+1];
    let mut dy = vec![0.0; n+1];

    let mut nm = n;
    if x < 1.0e-60 {
        for k in (0..=n).map(|k| k as usize) {
            sy[k] = -f64::INFINITY;
            dy[k] = f64::INFINITY;
        }
        return (sy, dy);
    }
    sy[0] = -f64::cos(x)/x;
    let mut f0 = sy[0];
    dy[0] = (f64::sin(x)+f64::cos(x)/x)/x;
    if n < 1 {
       return (sy, dy);
    }
    sy[1] = (sy[0]-f64::sin(x))/x;
    let mut f1 = sy[1];
    for k in (2..=n).map(|k| k as usize) {
       let f = (2.0*(k as f64)-1.0)*f1/x-f0;
       sy[k] = f;
       if f.abs() == f64::INFINITY {
            nm = k;
            break;
       }
       f0 = f1;
       f1 = f;
       nm = k;
    }
    for k in (1..=nm).map(|k| k as usize) {
        dy[k] = sy[k-1]-(k as f64 + 1.0)*sy[k]/x;
    }
    return (sy, dy);
}

fn msta2(x: f64, n: i32, mp: i32) -> i32 {
    let a0 = f64::abs(x);
    let hmp = 0.5*(mp as f64);
    let ejn = envj(n, a0);
    let mut obj = hmp+ejn;
    let mut n0 = n;
    if ejn <= hmp {
        obj = mp.into();
        n0 = ((1.1*a0) as i32) + 1;
    }
    let mut f0 = envj(n0, a0) - obj;
    let mut n1 = n0 + 5;
    let mut f1 = envj(n1, a0) - obj;
    for _iter in 0..20 {
        let nn = ((n1 as f64)-((n1-n0) as f64)/(1.0-f0/f1)) as i32;
        let f = envj(nn, a0) - obj;
        if (nn-n1).abs() < 1 {
            return nn + 10;
        }
        n0 = n1;
        f0 = f1;
        n1 = nn;
        f1 = f;
    }
    panic!("There is a problem with the spherical Bessel module.");
}

fn msta1(x: f64, mp: i32) -> i32 {
    let a0 = f64::abs(x);
    let mut n0 = ((1.1*a0) as i32) + 1;
    let mut f0 = envj(n0, a0)-(mp as f64);
    let mut n1 = n0 + 5;
    let mut f1 = envj(n1, a0)-(mp as f64);

    for _iter in 0..20 {
        let nn = ((n1 as f64)-((n1-n0) as f64)/(1.0-f0/f1)) as i32;
        let f = envj(nn, a0)-(mp as f64);
        if (nn-n1).abs() < 1 {
            return nn;
        }
        n0 = n1;
        f0 = f1;
        n1 = nn;
        f1 = f;
    } 
    panic!("There is a problem with the spherical Bessel module.");
}

fn envj(n: i32, x: f64) -> f64 {
    let n = n as f64;
    0.5*((6.28*n).log10())-n*((1.36*x/n).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use approx::assert_relative_eq;

    #[test]
    fn test_spherical_bessel_first_kind_exact() {
        // https://dlmf.nist.gov/10.49.E3
        let x_vec = vec![0.12, 1.23, 12.34, 123.45, 1234.5];
        for x in x_vec {
            assert_relative_eq!(
                (-1.0/x + 3.0/(x*x*x))*f64::sin(x) - 3.0/(x*x)*f64::cos(x), 
                (spherical_bessel_first_kind(2, x).0)[2], 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_first_kind_recurrence() {
        // https://dlmf.nist.gov/10.51.E1
        let n_vec = vec![1, 2, 3, 7, 12];
        let x = 0.12;
        for n in n_vec {
            let n = n as usize;
            assert_relative_eq!(
                (spherical_bessel_first_kind(n-1, x).0)[n-1] + (spherical_bessel_first_kind(n+1, x).0)[n+1],
                (2*n + 1) as f64 / x * (spherical_bessel_first_kind(n, x).0)[n], 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_first_kind_at_zero() {
        // https://dlmf.nist.gov/10.52.E1
        let n_vec = vec![0, 1, 2, 5, 10, 100];
        let x = 0.0;
        for n in n_vec {
            let n = n as usize;
            let mut correct_value = 0.0;
            if n == 0 {
                correct_value = 1.0;
            }
            assert_relative_eq!(
                (spherical_bessel_first_kind(n, x).0)[n],
                correct_value, 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_second_kind_exact() {
        // https://dlmf.nist.gov/10.49.E5
        let x_vec = vec![0.12, 1.23, 12.34, 123.45, 1234.5];
        for x in x_vec {
            println!("{}", x);
            assert_relative_eq!(
                (1.0/x - 3.0/(x*x*x))*f64::cos(x) - 3.0/(x*x)*f64::sin(x), 
                (spherical_bessel_second_kind(2, x).0)[2], 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_second_kind_recurrence() {
        // https://dlmf.nist.gov/10.51.E1
        let n_vec = vec![1, 2, 3, 7, 12];
        let x = 0.12;
        for n in n_vec {
            let n = n as usize;
            assert_relative_eq!(
                (spherical_bessel_second_kind(n-1, x).0)[n-1] + (spherical_bessel_second_kind(n+1, x).0)[n+1],
                (2*n + 1) as f64 / x * (spherical_bessel_second_kind(n, x).0)[n], 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_second_kind_at_zero() {
        // https://dlmf.nist.gov/10.52.E2
        let n_vec = vec![0, 1, 2, 5, 10, 100];
        let x = 0.0;
        for n in n_vec {
            let n = n as usize;
            assert_relative_eq!(
                (spherical_bessel_second_kind(n, x).0)[n],
                -f64::INFINITY, 
                max_relative=1e-10
            );
        }
    }

    #[test]
    fn test_spherical_bessel_cross_product_1() {
        // https://dlmf.nist.gov/10.50.E3
        let n_vec = vec![1, 5, 8];
        let x_vec = vec![0.1, 1.0, 10.0];
        for n in n_vec {
            let n = n as usize;
            for x in x_vec.clone() {
                assert_relative_eq!(
                    (spherical_bessel_first_kind(n+1, x).0)[n+1] * (spherical_bessel_second_kind(n, x).0)[n] -
                    (spherical_bessel_first_kind(n, x).0)[n] * (spherical_bessel_second_kind(n + 1, x).0)[n+1],
                    1.0/(x*x),
                    max_relative=1e-10
                );
            }
        }
    }

    #[test]
    fn test_spherical_bessel_cross_product_2() {
        // https://dlmf.nist.gov/10.50.E3
        let n_vec = vec![1, 5, 8];
        let x_vec = vec![0.1, 1.0, 10.0];
        for n in n_vec {
            let n = n as usize;
            for x in x_vec.clone() {
                assert_relative_eq!(
                    (spherical_bessel_first_kind(n+2, x).0)[n+2] * (spherical_bessel_second_kind(n, x).0)[n] -
                    (spherical_bessel_first_kind(n, x).0)[n] * (spherical_bessel_second_kind(n+2, x).0)[n+2],
                    (2.0*(n as f64)+3.0)/(x*x*x),
                    max_relative=1e-10
                );
            }
        }
    }

}