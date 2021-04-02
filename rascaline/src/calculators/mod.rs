use crate::descriptor::{Descriptor, Indexes, SamplesIndexes};
use crate::system::System;

#[cfg(test)]
mod tests_utils;

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

    /// Get the default set of samples for this Calculator
    fn samples(&self) -> Box<dyn SamplesIndexes>;
    /// Does this calculator compute gradients?
    fn compute_gradients(&self) -> bool;

    /// Check that the given indexes are valid feature indexes for this
    /// Calculator. This is used by `Calculator::compute_partial` to ensure
    /// only valid features are requested
    fn check_features(&self, indexes: &Indexes);
    /// Check that the given indexes are valid samples indexes for this
    /// Calculator. This is used by `Calculator::compute_partial` to ensure
    /// only valid samples are requested
    fn check_samples(&self, indexes: &Indexes, systems: &mut [Box<dyn System>]);

    /// Core implementation of the descriptor.
    ///
    /// This function should compute the descriptor only for samples in
    /// `descriptor.samples` and computing only features in
    /// `descriptor.features`. By default, these would correspond to the samples
    /// and features coming from `Descriptor::samples()` and
    /// `Descriptor::features()` respectively; but the user can request only a
    /// subset of them.
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor);
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;

pub mod soap;
pub use self::soap::{SphericalExpansion, SphericalExpansionParameters};
pub use self::soap::{SoapPowerSpectrum, PowerSpectrumParameters};
