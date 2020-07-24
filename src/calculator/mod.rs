use crate::system::System;
use crate::descriptor::Descriptor;

/// TODO: docs
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// calculators across the C API.
pub trait Calculator: std::panic::RefUnwindSafe {
    /// Get the name of this Calculator
    fn name(&self) -> String;
    /// Compute the descriptor for all the given systems and store it in `descriptor`
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor);
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;
