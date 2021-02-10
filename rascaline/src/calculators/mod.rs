use crate::descriptor::{Descriptor, Indexes, EnvironmentIndexes};
use crate::system::System;

/// TODO: docs
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// calculators across the C API.
pub trait CalculatorBase: std::panic::RefUnwindSafe {
    /// Get the name of this Calculator
    fn name(&self) -> String;

    /// Get the parameters used to create this Calculator as a JSON string
    fn get_parameters(&self) -> String;

    /// Get the names of features for this Calculator
    fn features_names(&self) -> Vec<&str>;
    /// Get the default set of features for this Calculator
    fn features(&self) -> Indexes;

    /// Get the default set of environments for this Calculator
    fn environments(&self) -> Box<dyn EnvironmentIndexes>;
    /// Does this environment compute gradients?
    fn compute_gradients(&self) -> bool;

    /// Check that the given indexes are valid feature indexes for this
    /// Calculator. This is used by `Calculator::compute_partial` to ensure
    /// only valid features are requested
    fn check_features(&self, indexes: &Indexes);
    /// Check that the given indexes are valid environment indexes for this
    /// Calculator. This is used by `Calculator::compute_partial` to ensure
    /// only valid environments are requested
    fn check_environments(&self, indexes: &Indexes, systems: &mut [&mut dyn System]);

    /// Core implementation of the descriptor.
    ///
    /// This function should compute the descriptor only for environments in
    /// `descriptor.environments` and computing only features in
    /// `descriptor.features`. By default, these would correspond to the
    /// environments and features coming from `Descriptor::environments` and
    /// `Descriptor::features` respectively; but this can be overrode to only
    /// compute them on a subset though `Descriptor::compute_partial`.
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor);
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;

pub mod soap;
pub use self::soap::{SphericalExpansion, SphericalExpansionParameters};
