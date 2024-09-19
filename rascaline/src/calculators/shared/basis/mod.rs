pub(crate) mod radial;

pub use self::radial::{SoapRadialBasis, LodeRadialBasis};

/// Possible Basis functions to use for the SOAP or LODE spherical expansion.
///
/// The basis is made of radial and angular parts, that can be combined in
/// various ways.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum SphericalExpansionBasis<RadialBasis> {
    /// A Tensor product basis, combining all possible radial basis functions
    /// with all possible angular basis functions.
    TensorProduct(TensorProductBasis<RadialBasis>)
}

impl<RadialBasis> SphericalExpansionBasis<RadialBasis> {
    pub fn angular_channels(&self) -> impl Iterator<Item=usize> {
        match self {
            SphericalExpansionBasis::TensorProduct(basis) => {
                return 0..=basis.max_angular;
            }
        }
    }
}


/// Information about tensor product bases
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TensorProductBasis<RadialBasis> {
    /// Maximal value (inclusive) of the angular moment (quantum number `l`) to
    /// use for the spherical harmonics basis functions
    pub max_angular: usize,
    /// Definition of the radial basis functions
    pub radial: RadialBasis,
    /// Accuracy for splining the radial integral. Using splines is typically
    /// faster than analytical implementations. If this is None, no splining is
    /// done.
    ///
    /// The number of control points in the spline is automatically determined
    /// to ensure the average absolute error is close to the requested accuracy.
    #[serde(default = "serde_default_spline_accuracy")]
    pub spline_accuracy: Option<f64>,
}

#[allow(clippy::unnecessary_wraps)]
fn serde_default_spline_accuracy() -> Option<f64> { Some(1e-8) }
