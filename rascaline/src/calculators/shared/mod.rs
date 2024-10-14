//! This module contains type definition shared between the SOAP and LODE
//! spherical expansions: atomic density, radial and angular basis, as well as
//! some parallelization helpers.

mod density;
pub use self::density::{Density, DensityKind, DensityScaling};

pub(crate) mod basis;
pub use self::basis::{SphericalExpansionBasis, TensorProductBasis, ExplicitBasis};
pub use self::basis::{SoapRadialBasis, LodeRadialBasis};

pub mod descriptors_by_systems;
