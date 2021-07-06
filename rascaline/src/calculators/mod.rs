use crate::descriptor::{Descriptor, Indexes, SamplesBuilder};

use crate::{Error, System};

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

    /// Get the default sample builder for this Calculator
    fn samples_builder(&self) -> Box<dyn SamplesBuilder>;
    /// Does this calculator compute gradients?
    fn compute_gradients(&self) -> bool;

    /// Check that the given indexes are valid feature indexes for this
    /// Calculator. This is used by to ensure only valid features are requested
    fn check_features(&self, indexes: &Indexes) -> Result<(), Error>;

    /// Check that the given indexes are valid samples indexes for this
    /// Calculator. This is used by to ensure only valid samples are requested
    ///
    /// The default implementation recompute the full set of samples using
    /// `Self::samples()`, and check that all requested samples are part of the
    /// full sample set.
    fn check_samples(&self, indexes: &Indexes, systems: &mut [Box<dyn System>]) -> Result<(), Error> {
        let builder = self.samples_builder();
        if indexes.names() != builder.names() {
            return Err(Error::InvalidParameter(format!(
                "invalid sample names for {}, expected [{}], got [{}]",
                self.name(),
                builder.names().join(", "),
                indexes.names().join(", "),
            )))
        }

        let allowed = builder.samples(systems)?;
        for value in indexes.iter() {
            if !allowed.contains(value) {
                return Err(Error::InvalidParameter(format!(
                    "{:?} is not a valid sample for {}", value, self.name()
                )))
            };
        }

        Ok(())
    }

    /// Core implementation of the descriptor.
    ///
    /// This function should compute the descriptor only for samples in
    /// `descriptor.samples` and computing only features in
    /// `descriptor.features`. By default, these would correspond to the samples
    /// and features coming from `Descriptor::samples()` and
    /// `Descriptor::features()` respectively; but the user can request only a
    /// subset of them.
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error>;
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;

pub mod soap;
pub use self::soap::{SphericalExpansion, SphericalExpansionParameters};
pub use self::soap::{SoapPowerSpectrum, PowerSpectrumParameters};
