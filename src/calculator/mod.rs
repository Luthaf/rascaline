use crate::descriptor::Descriptor;
use crate::system::System;
use crate::Error;

/// TODO: docs
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// calculators across the C API.
pub trait Calculator: std::panic::RefUnwindSafe {
    /// Get the name of this Calculator
    fn name(&self) -> String;
    /// Get the parameters used to create this Calculator in a string.
    ///
    /// Currently the string is formatted as JSON, but this could change in the
    /// future.
    fn parameters(&self) -> Result<String, Error>;
    /// Compute the descriptor for all the given systems and store it in `descriptor`
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor);
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;
