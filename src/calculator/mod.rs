use crate::system::System;
use crate::descriptor::Descriptor;

pub trait Calculator {
    /// Get the name of this Calculator
    fn name(&self) -> String;
    /// Compute the descriptor for all the given systems and store it in `descriptor`
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor);
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;

mod dummy_calculator;
pub use self::dummy_calculator::DummyCalculator;
