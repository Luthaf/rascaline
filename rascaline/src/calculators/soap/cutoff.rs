use crate::Error;

/// Definition of a local environment for SOAP calculations
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Cutoff {
    /// Radius of the spherical cutoff to use for atomic environments
    pub radius: f64,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub smoothing: Smoothing,
}

/// Possible values for the smoothing cutoff function
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum Smoothing {
    /// Shifted cosine smoothing function
    /// `f(r) = 1/2 * (1 + cos(π (r - cutoff + width) / width ))`
    ShiftedCosine {
        /// Width of the switching function
        width: f64,
    },
    /// Step smoothing function (i.e. no smoothing). This is 1 inside the cutoff
    /// and 0 outside, with a sharp step at the boundary.
    Step,
}

impl Cutoff {
    pub fn validate(&self) -> Result<(), Error> {
        match self.smoothing {
            Smoothing::Step => {},
            Smoothing::ShiftedCosine { width } => {
                if width <= 0.0 || !width.is_finite() {
                    return Err(Error::InvalidParameter(format!(
                        "expected positive width for shifted cosine cutoff function, got {}",
                        width
                    )));
                }
            }
        }
        return Ok(());
    }

    /// Evaluate the smoothing function at the distance `r`
    pub fn smoothing(&self, r: f64) -> f64 {
        match self.smoothing {
            Smoothing::Step => {
                if r >= self.radius { 0.0 } else { 1.0 }
            },
            Smoothing::ShiftedCosine { width } => {
                if r <= (self.radius - width) {
                    1.0
                } else if r >= self.radius {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - self.radius + width) / width;
                    0.5 * (1. + f64::cos(s))
                }
            }
        }
    }

    /// Evaluate the gradient of the smoothing function at the distance `r`
    pub fn smoothing_gradient(&self, r: f64) -> f64 {
        match self.smoothing {
            Smoothing::Step => 0.0,
            Smoothing::ShiftedCosine { width } => {
                if r <= (self.radius - width) || r >= self.radius {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - self.radius + width) / width;
                    return -0.5 * std::f64::consts::PI * f64::sin(s) / width;
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_smoothing() {
        let cutoff = Cutoff { radius: 4.0, smoothing: Smoothing::Step};

        assert_eq!(cutoff.smoothing(2.0), 1.0);
        assert_eq!(cutoff.smoothing(5.0), 0.0);

        assert_eq!(cutoff.smoothing_gradient(2.0), 0.0);
        assert_eq!(cutoff.smoothing_gradient(5.0), 0.0);
    }

    #[test]
    fn shifted_cosine() {
        let cutoff = Cutoff { radius: 4.0, smoothing: Smoothing::ShiftedCosine { width: 0.5 }};

        assert_eq!(cutoff.smoothing(2.0), 1.0);
        assert_eq!(cutoff.smoothing(3.5), 1.0);
        assert_eq!(cutoff.smoothing(3.8), 0.34549150281252683);
        assert_eq!(cutoff.smoothing(4.0), 0.0);
        assert_eq!(cutoff.smoothing(5.0), 0.0);

        assert_eq!(cutoff.smoothing_gradient(2.0), 0.0);
        assert_eq!(cutoff.smoothing_gradient(3.5), 0.0);
        assert_eq!(cutoff.smoothing_gradient(3.8), -2.987832164741557);
        assert_eq!(cutoff.smoothing_gradient(4.0), 0.0);
        assert_eq!(cutoff.smoothing_gradient(5.0), 0.0);
    }
}
