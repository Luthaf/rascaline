use metatensor::{TensorMap, Labels};

use crate::{Error, System};


/// Which gradients are we computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(self) struct GradientsOptions {
    pub positions: bool,
    pub cell: bool,
    pub strain: bool,
}

impl GradientsOptions {
    pub fn any(self) -> bool {
        return self.positions || self.cell || self.strain;
    }
}

/// The `CalculatorBase` trait is the interface shared by all calculator
/// implementations; and used by [`crate::Calculator`] to run the calculation.
///
/// This should not be used directly by end users, who should use the facilities
/// in [`crate::Calculator`] instead.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// calculators across the C API.
pub trait CalculatorBase: std::panic::RefUnwindSafe {
    /// Get the name of this Calculator
    fn name(&self) -> String;

    /// Get the parameters used to create this Calculator as a JSON string
    fn parameters(&self) -> String;

    /// Get the all radial cutoffs used by this Calculator's neighbors lists
    /// (which can be an empty list)
    fn cutoffs(&self) -> &[f64];

    /// Get the set of keys for this calculator and the given systems
    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error>;

    /// Get the names used for sample labels by this calculator
    fn sample_names(&self) -> Vec<&str>;

    /// Get the full list of samples this calculator would create for the given
    /// systems. This function should return one set of samples for each key.
    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error>;

    /// Can this calculator compute gradients with respect to the `parameter`?
    /// Right now, `parameter` can be either `"positions"`, `"strain"` or
    /// `"cell"`.
    fn supports_gradient(&self, parameter: &str) -> bool;

    /// Get the samples for gradients with respect to positions, corresponding
    /// the given values samples.
    ///
    /// The `samples` slice contains one set of samples for each key.
    ///
    /// If the gradients with respect to positions are not available, this
    /// function should return an error.
    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error>;

    /// Get the components this calculator computes for each key.
    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>>;

    /// Get the names used for property labels by this calculator
    fn property_names(&self) -> Vec<&str>;

    /// Get the properties this calculator computes for each key.
    fn properties(&self, keys: &Labels) -> Vec<Labels>;

    /// Actually run the calculation.
    ///
    /// This function is given a pre-allocated descriptor, filled with zeros.
    /// The samples/properties in each blocks might not match the values
    /// returned by [`CalculatorBase::samples`] and
    /// [`CalculatorBase::properties`]: instead they will only contain the
    /// values that where requested by the end user.
    ///
    /// Gradients (with respect to positions or cell) are allocated in each
    /// block if they are supported according to
    /// [`CalculatorBase::supports_gradient`], and the users requested them as
    /// part of the calculation options.
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error>;
}


#[cfg(test)]
pub(crate) mod tests_utils;

mod atomic_composition;
pub use self::atomic_composition::AtomicComposition;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod neighbor_list;
pub use self::neighbor_list::NeighborList;

mod radial_basis;
pub use self::radial_basis::{RadialBasis, GtoRadialBasis};

mod descriptors_by_systems;
pub(crate) use self::descriptors_by_systems::{array_mut_for_system, split_tensor_map_by_system};

pub mod soap;
pub use self::soap::{SphericalExpansionByPair, SphericalExpansionParameters};
pub use self::soap::SphericalExpansion;
pub use self::soap::{SoapPowerSpectrum, PowerSpectrumParameters};
pub use self::soap::{SoapRadialSpectrum, RadialSpectrumParameters};

pub mod lode;
pub use self::lode::{LodeSphericalExpansion, LodeSphericalExpansionParameters};
