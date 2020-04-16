use crate::system::System;
use crate::descriptor::Descriptor;

pub trait Calculator {
    /// Get the name of this Calculator
    fn name(&self) -> String;
    /// Compute the descriptor for all the given systems
    fn compute(&mut self, systems: &mut [&mut dyn System]) -> Descriptor;
}

mod sorted_distances;
pub use self::sorted_distances::SortedDistances;
