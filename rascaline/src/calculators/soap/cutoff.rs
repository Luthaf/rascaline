use crate::Error;

/// Possible values for the smoothing cutoff function
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub enum CutoffFunction {
    /// Step function, 1 if `r < cutoff` and 0 if `r >= cutoff`
    Step{},
    /// Shifted cosine switching function
    /// `f(r) = 1/2 * (1 + cos(π (r - cutoff + width) / width ))`
    ShiftedCosine {
        width: f64,
    },
}

impl CutoffFunction {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            CutoffFunction::Step {} => {},
            CutoffFunction::ShiftedCosine { width } => {
                if *width <= 0.0 {
                    return Err(Error::InvalidParameter(format!(
                        "expected positive width for shifted cosine cutoff function, got {}",
                        width
                    )));
                }
            }
        }
        return Ok(());
    }

    /// Evaluate the cutoff function at the distance `r` for the given `cutoff`
    pub fn compute(&self, r: f64, cutoff: f64) -> f64 {
        match self {
            CutoffFunction::Step{} => {
                if r >= cutoff { 0.0 } else { 1.0 }
            },
            CutoffFunction::ShiftedCosine { width } => {
                if r <= (cutoff - width) {
                    1.0
                } else if r >= cutoff {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - cutoff + width) / width;
                    0.5 * (1. + f64::cos(s))
                }
            }
        }
    }

    /// Evaluate the derivative of the cutoff function at the distance `r` for the
    /// given `cutoff`
    pub fn derivative(&self, r: f64, cutoff: f64) -> f64 {
        match self {
            CutoffFunction::Step{} => 0.0,
            CutoffFunction::ShiftedCosine { width } => {
                if r <= (cutoff - width) || r >= cutoff {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - cutoff + width) / width;
                    return -0.5 * std::f64::consts::PI * f64::sin(s) / width;
                }
            }
        }
    }
}

/// Implemented options for radial scaling of the atomic density around an atom
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub enum RadialScaling {
    /// No radial scaling
    None {},
    /// Use the radial scaling functional introduced in <https://doi.org/10.1039/C8CP05921G>:
    ///
    /// `f(r) = rate / (rate + (r / scale) ^ exponent)`
    Willatt2018 {
        scale: f64,
        rate: f64,
        exponent: i32,
    },
}

impl Default for RadialScaling {
    fn default() -> RadialScaling {
        RadialScaling::None {}
    }
}

impl RadialScaling {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            RadialScaling::None {} => {},
            RadialScaling::Willatt2018 { scale, rate, exponent } => {
                if *scale <= 0.0 {
                    return Err(Error::InvalidParameter(format!(
                        "expected positive scale for Willatt2018 radial scaling function, got {}",
                        scale
                    )));
                }

                if *rate <= 0.0 {
                    return Err(Error::InvalidParameter(format!(
                        "expected positive rate for Willatt2018 radial scaling function, got {}",
                        rate
                    )));
                }

                if *exponent <= 0 {
                    return Err(Error::InvalidParameter(format!(
                        "expected positive exponent for Willatt2018 radial scaling function, got {}",
                        exponent
                    )));
                }
            }
        }
        return Ok(());
    }

    /// Evaluate the radial scaling function at the distance `r`
    pub fn compute(&self, r: f64) -> f64 {
        match self {
            RadialScaling::None {} => 1.0,
            RadialScaling::Willatt2018 { rate, scale, exponent } => {
                rate / (rate + (r / scale).powi(*exponent))
            }
        }
    }

    /// Evaluate the derivative of the radial scaling function at the distance `r`
    pub fn derivative(&self, r: f64) -> f64 {
        match self {
            RadialScaling::None {} => 0.0,
            RadialScaling::Willatt2018 { scale, rate, exponent } => {
                let rs = r / scale;
                let rs_m1 = rs.powi(exponent - 1);
                let rs_m = rs * rs_m1;
                let factor = - rate * (*exponent as f64) / scale;

                factor * rs_m1 / ((rate + rs_m) * (rate + rs_m))
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn step() {
        let function = CutoffFunction::Step{};
        let cutoff = 4.0;

        assert_eq!(function.compute(2.0, cutoff), 1.0);
        assert_eq!(function.compute(5.0, cutoff), 0.0);
    }

    #[test]
    fn step_gradient() {
        let function = CutoffFunction::Step{};
        let cutoff = 4.0;

        assert_eq!(function.derivative(2.0, cutoff), 0.0);
        assert_eq!(function.derivative(5.0, cutoff), 0.0);
    }

    #[test]
    fn shifted_cosine() {
        let function = CutoffFunction::ShiftedCosine { width: 0.5 };
        let cutoff = 4.0;

        assert_eq!(function.compute(2.0, cutoff), 1.0);
        assert_eq!(function.compute(3.5, cutoff), 1.0);
        assert_eq!(function.compute(3.8, cutoff), 0.34549150281252683);
        assert_eq!(function.compute(4.0, cutoff), 0.0);
        assert_eq!(function.compute(5.0, cutoff), 0.0);
    }

    #[test]
    fn shifted_cosine_gradient() {
        let function = CutoffFunction::ShiftedCosine { width: 0.5 };
        let cutoff = 4.0;

        assert_eq!(function.derivative(2.0, cutoff), 0.0);
        assert_eq!(function.derivative(3.5, cutoff), 0.0);
        assert_eq!(function.derivative(3.8, cutoff), -2.987832164741557);
        assert_eq!(function.derivative(4.0, cutoff), 0.0);
        assert_eq!(function.derivative(5.0, cutoff), 0.0);
    }
}
