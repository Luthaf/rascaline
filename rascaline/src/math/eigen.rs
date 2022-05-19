// Eigen decomposition of symmetric matrix. Adapted from
// https://github.com/xasmx/rust-la, which is a Rust port of the JAMA implementation
// https://en.wikipedia.org/wiki/JAMA_(numerical_linear_algebra_library)

use ndarray::{Array2, Array1};

/// Eigendecomposition of a real symmetric matrix into eigenvalues and eigenvectors
#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    /// Eigenvalues of the input matrix, sorted in increasing order
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors of the input matrix
    pub eigenvectors: Array2<f64>,
}

impl SymmetricEigen {
    // Symmetric Householder reduction to tridiagonal form.
    //
    //  This is derived from the Algol procedures tred2 by Bowdler, Martin,
    //  Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear
    //  Algebra, and the corresponding Fortran subroutine in EISPACK.
    #[allow(clippy::needless_range_loop)]
    fn tridiagonalize(eval: &mut Array1<f64>, evec: &mut Array2<f64>, work: &mut [f64]) {
        let n = eval.len();
        debug_assert_eq!(work.len(), n);
        debug_assert_eq!(evec.len(), n * n);

        for j in 0..n {
            eval[j] = evec[[(n - 1), j]];
        }

        // Householder reduction to tridiagonal form.
        for i in (1..n).rev() {
            // Scale to avoid under/overflow.
            let mut scale = 0.0;
            let mut h = 0.0;
            for k in 0..i {
                scale += f64::abs(eval[k]);
            }
            if scale == 0.0 {
                work[i] = eval[i - 1];
                for j in 0..i {
                    eval[j] = evec[[(i - 1), j]];
                    evec[[i, j]] = 0.0;
                    evec[[j, i]] = 0.0;
                }
            } else {
                // Generate Householder vector.
                for k in 0..i {
                    eval[k] /= scale;
                    h += eval[k] * eval[k];
                }
                let mut f = eval[i - 1];
                let mut g = h.sqrt();
                if f > 0.0 {
                    g = -g;
                }
                work[i] = scale * g;
                h -= f * g;
                eval[i - 1] = f - g;
                for j in 0..i {
                    work[j] = 0.0;
                }

                // Apply similarity transformation to remaining columns.
                for j in 0..i {
                    f = eval[j];
                    evec[[j, i]] = f;
                    g = work[j] + evec[[j, j]] * f;
                    for k in (j + 1)..i {
                        g += evec[[k, j]] * eval[k];
                        work[k] += evec[[k, j]] * f;
                    }
                    work[j] = g;
                }
                f = 0.0;
                for j in 0..i {
                    work[j] /= h;
                    f += work[j] * eval[j];
                }
                let hh = f / (h + h);
                for j in 0..i {
                    work[j] -= hh * eval[j];
                }
                for j in 0..i {
                    f = eval[j];
                    g = work[j];
                    for k in j..i {
                        let orig_val = evec[[k, j]];
                        evec[[k, j]] = orig_val - (f * work[k] + g * eval[k]);
                    }
                    eval[j] = evec[[(i - 1), j]];
                    evec[[i, j]] = 0.0;
                }
            }
            eval[i] = h;
        }

        // Accumulate transformations.
        for i in 0..(n - 1) {
            let orig_val = evec[[i, i]];
            evec[[(n - 1), i]] = orig_val;
            evec[[i, i]] = 1.0;
            let h = eval[i + 1];
            if h != 0.0 {
                for k in 0..(i + 1) {
                    eval[k] = evec[[k, (i + 1)]] / h;
                }
                for j in 0..(i + 1) {
                    let mut g = 0.0;
                    for k in 0..(i + 1) {
                        g += evec[[k, (i + 1)]] * evec[[k, j]];
                    }
                    for k in 0..(i + 1) {
                        let orig_val = evec[[k, j]];
                        evec[[k, j]] = orig_val - g * eval[k];
                    }
                }
            }
            for k in 0..(i + 1) {
                evec[[k, (i + 1)]] = 0.0;
            }
        }
        for j in 0..n {
            eval[j] = evec[[(n - 1), j]];
            evec[[(n - 1), j]] = 0.0;
        }
        evec[[(n - 1), (n - 1)]] = 1.0;
        work[0] = 0.0;
    }

    // Symmetric tridiagonal QL algorithm.
    //
    // This is derived from the Algol procedures tql2, by Bowdler, Martin,
    // Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear
    // Algebra, and the corresponding Fortran subroutine in EISPACK.
    #[allow(clippy::many_single_char_names)]
    fn tridiagonal_ql(eval: &mut Array1<f64>, evec: &mut Array2<f64>, work: &mut [f64]) {
        let n = eval.len();
        debug_assert_eq!(work.len(), n);
        debug_assert_eq!(evec.len(), n * n);

        for i in 1..n {
            work[i - 1] = work[i];
        }
        work[n - 1] = 0.0;

        let mut f = 0.0;
        let mut tst1 = 0.0f64;
        for l in 0..n {
            // Find small subdiagonal element
            tst1 = tst1.max(f64::abs(eval[l]) + f64::abs(work[l]));
            let mut m = l;
            while m < n {
                if f64::abs(work[m]) <= (f64::EPSILON * tst1) {
                    break;
                }
                m += 1;
            }

            // If m == l, d[l] is an eigenvalue, otherwise, iterate.
            if m > l {
                loop {
                    // Compute implicit shift
                    let mut g = eval[l];
                    let tmp = 2.0;
                    let mut p = (eval[l + 1] - g) / (tmp * work[l]);
                    let mut r = f64::hypot(p, 1.0);
                    if p < 0.0 {
                        r = -r;
                    }
                    eval[l] = work[l] / (p + r);
                    eval[l + 1] = work[l] * (p + r);
                    let eval_lp1 = eval[l + 1];
                    let mut h = g - eval[l];
                    for i in (l + 2)..n {
                        eval[i] -= h;
                    }
                    f += h;

                    // Implicit QL transformation.
                    p = eval[m];
                    let mut c = 1.0;
                    let mut c2 = c;
                    let mut c3 = c;
                    let work_lp1 = work[l + 1];
                    let mut s = 0.0;
                    let mut s2 = 0.0;
                    for i in (l..m).rev() {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * work[i];
                        h = c * p;
                        r = f64::hypot(p, work[i]);
                        work[i + 1] = s * r;
                        s = work[i] / r;
                        c = p / r;
                        p = c * eval[i] - s * g;
                        eval[i + 1] = h + s * (c * g + s * eval[i]);

                        // Accumulate transformation.
                        for k in 0..n {
                            h = evec[[k, (i + 1)]];
                            evec[[k, (i + 1)]] = s * evec[[k, i]] + c * h;
                            evec[[k, i]] = c * evec[[k, i]] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * work_lp1 * work[l] / eval_lp1;
                    work[l] = s * p;
                    eval[l] = c * p;

                    // Check for convergence.
                    if f64::abs(work[l]) <= (f64::EPSILON * tst1) {
                        break;
                    }
                }
            }
            eval[l] += f;
            work[l] = 0.0;
        }

        // Bubble sort eigenvalues and corresponding vectors.
        for i in 0..(n - 1) {
            let mut k = i;
            let mut p = eval[i];
            for j in (i + 1)..n {
                if eval[j] < p {
                    k = j;
                    p = eval[j];
                }
            }
            if k != i {
                // Swap columns k and i of the diagonal and v.
                eval[k] = eval[i];
                eval[i] = p;
                for j in 0..n {
                    p = evec[[j, i]];
                    evec[[j, i]] = evec[[j, k]];
                    evec[[j, k]] = p;
                }
            }
        }
    }

    /// Compute the eigendecomposition of a symmetric real matrix
    #[allow(clippy::float_cmp)]
    pub fn new(matrix: Array2<f64>) -> SymmetricEigen {
        assert_eq!(matrix.nrows(), matrix.ncols());

        let n = matrix.ncols();
        for i in 0..n {
            for j in i..n {
                debug_assert_eq!(matrix[[i, j]], matrix[[j, i]], "matrix is not symmetric");
            }
        }

        let mut eigenvectors = matrix;
        let mut eigenvalues = Array1::from_elem(n, 0.0);

        let mut work = vec![0.0; n];

        SymmetricEigen::tridiagonalize(&mut eigenvalues, &mut eigenvectors, &mut work);
        SymmetricEigen::tridiagonal_ql(&mut eigenvalues, &mut eigenvectors, &mut work);

        SymmetricEigen {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Recreate the input matrix from the eigenvalues and eigenvectors
    pub fn recompose(&self) -> Array2<f64> {
        let result = self.eigenvectors.dot(&Array2::from_diag(&self.eigenvalues));
        return result.dot(&self.eigenvectors.t());
    }
}
