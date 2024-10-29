mod gto;
pub use self::gto::GtoRadialBasis;

mod tabulated;
pub use self::tabulated::{Tabulated, LodeTabulated};


/// The different kinds of radial basis supported by SOAP calculators
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum SoapRadialBasis {
    /// Use a radial basis similar to Gaussian-Type Orbitals.
    ///
    /// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
    /// = cutoff * \sqrt{n} / n_max`
    Gto {
        /// Maximal value of `n` to include in the radial basis function
        /// definition. The overall basis will have `max_radial + 1` basis
        /// functions, indexed from `0` to `max_radial` (inclusive).
        max_radial: usize,

        #[doc(hidden)]
        ///
        radius: Option<f64>,
    },
    /// Use pre-tabulated radial basis.
    ///
    /// This enables the calculation of the overall radial integral using
    /// user-defined splines.
    ///
    /// The easiest way to create such a tabulated basis is the corresponding
    /// functions in rascaline's Python API.
    #[schemars(with = "tabulated::TabulatedSerde")]
    Tabulated(Tabulated)
}

impl SoapRadialBasis {
    /// Get the size (number of basis function) for the current basis
    pub fn size(&self) -> usize {
        match self {
            SoapRadialBasis::Gto { max_radial, .. } => max_radial + 1,
            SoapRadialBasis::Tabulated(tabulated) => tabulated.size(),
        }
    }
}


/// The different kinds of radial basis supported LODE calculators
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum LodeRadialBasis {
    /// Use a radial basis similar to Gaussian-Type Orbitals.
    ///
    /// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
    /// = radius * \sqrt{n} / n_max`
    Gto {
        /// Maximal value of `n` to include in the radial basis function
        /// definition. The overall basis will have `max_radial + 1` basis
        /// functions, indexed from `0` to `max_radial` (inclusive).
        max_radial: usize,
        /// Radius of the Gto basis, i.e. how far should the local LODE field be
        /// integrated.
        radius: f64,
    },
    /// Use pre-tabulated radial basis.
    ///
    /// This enables the calculation of the overall radial integral using
    /// user-defined splines.
    ///
    /// The easiest way to create such a tabulated basis is the corresponding
    /// functions in rascaline's Python API.
    #[schemars(with = "tabulated::LodeTabulatedSerde")]
    Tabulated(LodeTabulated)
}

impl LodeRadialBasis {
    /// Get the size (number of basis function) for the current basis
    pub fn size(&self) -> usize {
        match self {
            LodeRadialBasis::Gto { max_radial, .. } => max_radial + 1,
            LodeRadialBasis::Tabulated(tabulated) => tabulated.size(),
        }
    }
}
