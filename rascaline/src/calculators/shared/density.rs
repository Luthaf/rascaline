use crate::Error;


/// Definition of the (atomic) density to expand on a basis
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct Density {
    #[serde(flatten)] // because of this flatten, we can not use deny_unknown_fields
    pub kind: DensityKind,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    #[serde(default)]
    pub scaling: Option<DensityScaling>,
    /// Weight of the central atom contribution to the density. If `1` the
    /// center atom contribution is weighted the same as any other contribution.
    /// If `0` the central atom does not contribute to the density at all.
    #[serde(default = "serde_default_center_atom_weight")]
    pub center_atom_weight: f64,
}

fn serde_default_center_atom_weight() -> f64 {
    return 1.0;
}


/// Different available kinds of atomic density to use in rascaline
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum DensityKind {
    /// Dirac delta atomic density
    DiracDelta,
    /// Gaussian atomic density `exp(-r^2/width^2)`
    Gaussian {
        /// Width of the gaussian, the same width is used for all atoms
        width: f64
    },
    /// Long-range Gaussian density `exp(-r^2/width^2) / r^exponent`
    ///
    /// LODE spherical expansion is currently implemented only for
    /// `potential_exponent < 10`. Some exponents can be connected to SOAP or
    /// physics-based quantities: p=0 uses the same Gaussian densities as SOAP,
    /// p=1 uses 1/r Coulomb-like densities, p=6 uses 1/r^6 dispersion-like
    /// densities.
    LongRangeGaussian {
        /// Width of the gaussian, the same width is used for all atoms
        width: f64,
        /// Exponent of the density
        exponent: usize
    },
}

/// Implemented options for radial scaling of the atomic density around an atom
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum DensityScaling {
    /// Use a long-range algebraic decay and smooth behavior at `r → 0` as
    /// introduced in <https://doi.org/10.1039/C8CP05921G>:
    /// `f(r) = rate / (rate + (r / scale) ^ exponent)`
    Willatt2018 {
        /// see in the formula
        scale: f64,
        /// see in the formula
        rate: f64,
        /// see in the formula
        exponent: f64,
    },
}


impl DensityScaling {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            DensityScaling::Willatt2018 { scale, rate, exponent } => {
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

                if *exponent <= 0.0 {
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
            DensityScaling::Willatt2018 { rate, scale, exponent } => {
                rate / (rate + (r / scale).powf(*exponent))
            }
        }
    }

    /// Evaluate the gradient of the radial scaling function at the distance `r`
    pub fn gradient(&self, r: f64) -> f64 {
        match self {
            DensityScaling::Willatt2018 { scale, rate, exponent } => {
                let rs = r / scale;
                let rs_m1 = rs.powf(exponent - 1.0);
                let rs_m = rs * rs_m1;
                let factor = - rate * exponent / scale;

                factor * rs_m1 / ((rate + rs_m) * (rate + rs_m))
            }
        }
    }
}
