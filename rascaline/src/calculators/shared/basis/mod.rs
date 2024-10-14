pub(crate) mod radial;

use std::collections::BTreeMap;

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
    /// This defines a tensor product basis, combining all possible radial basis
    /// functions with all possible angular basis functions.
    TensorProduct(TensorProductBasis<RadialBasis>),
    /// This defines an explicit basis, where only a specific subset of angular
    /// basis can be used, and every angular basis can use a different radial
    /// basis.
    Explicit(ExplicitBasis<RadialBasis>),
}

impl<RadialBasis> SphericalExpansionBasis<RadialBasis> {
    pub fn angular_channels(&self) -> Vec<usize> {
        match self {
            SphericalExpansionBasis::TensorProduct(basis) => {
                return (0..=basis.max_angular).collect();
            }
            SphericalExpansionBasis::Explicit(basis) => {
                return basis.by_angular.keys().copied().collect();
            }
        }
    }
}

#[allow(clippy::unnecessary_wraps)]
fn serde_default_spline_accuracy() -> Option<f64> { Some(1e-8) }

/// Information about "tensor product" spherical expansion basis functions
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

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
// work around https://github.com/serde-rs/serde/issues/1183
#[serde(try_from = "BTreeMap<String, RadialBasis>")]
pub struct ByAngular<RadialBasis>(BTreeMap<usize, RadialBasis>);

impl<RadialBasis> std::ops::Deref for ByAngular<RadialBasis> {
    type Target = BTreeMap<usize, RadialBasis>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<RadialBasis> TryFrom<BTreeMap<String, RadialBasis>> for ByAngular<RadialBasis> {
    type Error = <usize as std::str::FromStr>::Err;

    fn try_from(value: BTreeMap<String, RadialBasis>) -> Result<Self, Self::Error> {
        let mut result = BTreeMap::new();
        for (angular, radial) in value {
            let angular: usize = angular.parse()?;
            result.insert(angular, radial);
        }
        Ok(ByAngular(result))
    }
}

impl<RadialBasis> From<BTreeMap<usize, RadialBasis>> for ByAngular<RadialBasis> {
    fn from(value: BTreeMap<usize, RadialBasis>) -> Self {
        ByAngular(value)
    }
}

/// Information about "explicit" spherical expansion basis functions
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExplicitBasis<RadialBasis> {
    /// A map of radial basis to use for the specified angular channels.
    ///
    /// Only angular channels included in this map will be included in the
    /// output. Different angular channels are allowed to use completely
    /// different radial basis functions.
    #[schemars(extend("x-key-type" = "integer"))]
    #[schemars(with = "BTreeMap<usize, RadialBasis>")]
    pub by_angular: ByAngular<RadialBasis>,
    /// Accuracy for splining the radial integral. Using splines is typically
    /// faster than analytical implementations. If this is None, no splining is
    /// done.
    ///
    /// The number of control points in the spline is automatically determined
    /// to ensure the average absolute error is close to the requested accuracy.
    #[serde(default = "serde_default_spline_accuracy")]
    pub spline_accuracy: Option<f64>,
}
