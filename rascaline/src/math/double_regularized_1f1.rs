use ndarray::ArrayViewMut1;

use super::{hyp1f1, gamma};

/// Compute `G(a, b, z) = Gamma(a) / Gamma(b) 1F1(a, b, z)` for
/// `a = 1/2 (n + l + 3)` and `b = l + 3/2` using recursion relations between
/// the value/gradients of this function for `l` and `l + 2`.
///
/// This is similar (but not the exact same) to the G function defined in
/// appendix A in <https://doi.org/10.1063/5.0044689>.
///
/// The function is called "double regularized 1F1" by reference to the
/// "regularized 1F1" function (i.e. `1F1(a, b, z) / Gamma(b)`)
#[derive(Clone, Copy, Debug)]
pub struct DoubleRegularized1F1 {
    pub max_angular: usize,
}

impl DoubleRegularized1F1 {
    /// Compute `Gamma(a) / Gamma(b) 1F1(a, b, z)` for all l and a given `n`
    pub fn compute(self, z: f64, n: usize, values: ArrayViewMut1<f64>, gradients: Option<ArrayViewMut1<f64>>) {
        debug_assert_eq!(values.len(), self.max_angular + 1);
        if let Some(ref gradients) = gradients {
            debug_assert_eq!(gradients.len(), self.max_angular + 1);
        }

        if self.max_angular < 3 {
            self.direct(z, n, values, gradients);
        } else {
            self.recursive(z, n, values, gradients);
        }
    }

    /// Direct evaluation of the G function
    fn direct(self, z: f64, n: usize, mut values: ArrayViewMut1<f64>, mut gradients: Option<ArrayViewMut1<f64>>) {
        for l in 0..=self.max_angular {
            let (a, b) = get_ab(l, n);
            let ratio = gamma(a) / gamma(b);

            values[l] = ratio * hyp1f1(a, b, z);
            if let Some(ref mut gradients) = gradients {
                gradients[l] = ratio * hyp1f1_derivative(a, b, z);
            }
        }
    }

    /// Recursive evaluation of the G function
    ///
    /// The recursion relations are derived from "Abramowitz and Stegun",
    /// rewriting equations 13.4.8, 13.4.10, 13.4.12, and 13.4.14 for G instead
    /// of M/1F1; and realizing that if G(a, b) is G(l); then G(a + 1, b + 2) is
    /// G(l+2).
    ///
    /// We end up with the following recurrence relations:
    ///
    /// - G'(l) = (b + 1) G(l + 2) + z G'(l + 2)
    /// - G(l) = b/a G'(l) + z (a - b)/a G(l + 2)
    ///
    /// Since the relation have a step of 2 for l, we initialize the recurrence
    /// by evaluating G for `l_max` and `l_max - 1`, and then propagate the
    /// values downward.
    #[allow(clippy::many_single_char_names)]
    fn recursive(self, z: f64, n: usize, mut values: ArrayViewMut1<f64>, mut gradients: Option<ArrayViewMut1<f64>>) {
        debug_assert!(self.max_angular >= 3);

        // initialize the values at l_max
        let mut l = self.max_angular;
        let (a, b) = get_ab(l, n);
        let ratio = gamma(a) / gamma(b);

        let mut g_l2 = ratio * hyp1f1(a, b, z);
        let mut grad_g_l2 = ratio * hyp1f1_derivative(a, b, z);
        values[l] = g_l2;
        if let Some(ref mut gradients) = gradients {
            gradients[l] = grad_g_l2;
        }

        // initialize the values at (l_max - 1)
        l -= 1;
        let (a, b) = get_ab(l, n);
        let ratio = gamma(a) / gamma(b);

        let mut g_l1 = ratio * hyp1f1(a, b, z);
        let mut grad_g_l1 = ratio * hyp1f1_derivative(a, b, z);
        values[l] = g_l1;
        if let Some(ref mut gradients) = gradients {
            gradients[l] = grad_g_l1;
        }

        let g_recursive_step = |a, b, g_l2, grad_g_l2| {
            let grad_g_l = (b + 1.0) * g_l2 + z * grad_g_l2;
            let g_l = (a - b) / a * z * g_l2 + b / a * grad_g_l;
            return (g_l, grad_g_l);
        };

        // do the recursion for all other l values
        l = self.max_angular;
        while l > 2 {
            l -= 2;
            let (a, b) = get_ab(l, n);
            let (new_value, new_grad) = g_recursive_step(a, b, g_l2, grad_g_l2);
            g_l2 = new_value;
            grad_g_l2 = new_grad;

            values[l] = g_l2;
            if let Some(ref mut gradients) = gradients {
                gradients[l] = grad_g_l2;
            }

            let (a, b) = get_ab(l - 1, n);
            let (new_value, new_grad) = g_recursive_step(a, b, g_l1, grad_g_l1);
            g_l1 = new_value;
            grad_g_l1 = new_grad;

            values[l - 1] = g_l1;
            if let Some(ref mut gradients) = gradients {
                gradients[l - 1] = grad_g_l1;
            }
        }

        // makes sure l == 0 is taken care of
        if self.max_angular % 2 == 0 {
            let (a, b) = get_ab(0, n);
            let (new_value, new_grad) = g_recursive_step(a, b, g_l2, grad_g_l2);
            g_l2 = new_value;
            grad_g_l2 = new_grad;

            values[0] = g_l2;
            if let Some(ref mut gradients) = gradients {
                gradients[0] = grad_g_l2;
            }
        }
    }
}

#[inline]
fn get_ab(l: usize, n: usize) -> (f64, f64) {
    return (0.5 * (n + l + 3) as f64, l as f64 + 1.5);
}


fn hyp1f1_derivative(a: f64, b: f64, x: f64) -> f64 {
    a / b * hyp1f1(a + 1.0, b + 1.0, x)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use approx::assert_relative_eq;

    #[test]
    fn direct_recursive_agree() {
        for &n in &[1, 5, 10, 18] {
            for &max_angular in &[8, 15] { //&[3, 8, 15] {

                let dr_1f1 = DoubleRegularized1F1{ max_angular };
                let mut direct = Array1::from_elem(max_angular + 1, 0.0);
                let mut recursive = Array1::from_elem(max_angular + 1, 0.0);

                for &z in &[-200.0, -10.0, -1.1, -1e-2, 0.2, 1.5, 10.0, 40.0, 523.0] {
                    dr_1f1.direct(z, n, direct.view_mut(), None);
                    dr_1f1.recursive(z, n, recursive.view_mut(), None);

                    assert_relative_eq!(
                        direct, recursive, max_relative=1e-9,
                    );
                }
            }
        }
    }


    #[test]
    fn finite_differences() {
        let delta = 1e-6;

        for &n in &[1, 5, 10, 18] {
            for &max_angular in &[0, 2, 8, 15] {

                let dr_1f1 = DoubleRegularized1F1{ max_angular };
                let mut values = Array1::from_elem(max_angular + 1, 0.0);
                let mut values_delta = Array1::from_elem(max_angular + 1, 0.0);
                let mut gradients = Array1::from_elem(max_angular + 1, 0.0);

                for &z in &[-200.0, -10.0, -1.1, -1e-2, 0.2, 1.5, 10.0, 40.0, 523.0] {
                    dr_1f1.compute(z, n, values.view_mut(), Some(gradients.view_mut()));
                    dr_1f1.compute(z + delta, n, values_delta.view_mut(), None);

                    let finite_difference = (&values_delta - &values) / delta;

                    assert_relative_eq!(
                        gradients, finite_difference, epsilon=delta, max_relative=1e-4,
                    );
                }
            }
        }
    }
}
