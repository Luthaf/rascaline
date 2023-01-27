
#[inline]
pub fn eval_polynomial(x: f64, coefficients: &[f64]) -> f64 {
    let mut result = coefficients[0];

    for c in &coefficients[1..] {
        result = result * x + c;
    }

    return result;
}


/// Same as `eval_polynomial`, but setting the coefficient of x^(n + 1) to 1.
pub fn eval_polynomial_1(x: f64, coefficients: &[f64]) -> f64 {
    let mut result = x + coefficients[0];

    for c in &coefficients[1..] {
        result = result * x + c;
    }

    return result;
}
