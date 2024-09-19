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
    /// Smeared power law density, that behaves like `1 / r^p` as `r` goes to
    /// infinity, while removing any singularity at `r=0` and ensuring the
    /// density is differentiable everywhere.
    ///
    /// The density functional form is `f(r) = 1 / Γ(p/2) * γ(p/2, r^2/(2 σ^2))
    /// / r^p`, with σ the smearing width, Γ the Gamma function and γ the lower
    /// incomplete gamma function.
    ///
    /// For more information about the derivation of this density, see
    /// <https://doi.org/10.1021/acs.jpclett.3c02375> and section D of the
    /// supplementary information.
    SmearedPowerLaw {
        /// Smearing width of the density (`σ`)
        smearing: f64,
        /// Exponent of the density (`p`)
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
