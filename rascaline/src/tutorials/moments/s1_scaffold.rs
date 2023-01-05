// [imports]
use equistore::{Labels, TensorMap, LabelsBuilder};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsSpeciesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, SpeciesFilter};
use crate::calculators::CalculatorBase;
// [imports]

// [struct]
#[derive(Clone, Debug)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
}
// [struct]

// [impl]
impl CalculatorBase for GeometricMoments {
    fn name(&self) -> String {
        todo!()
    }

    fn parameters(&self) -> String {
        todo!()
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        todo!()
    }

    fn samples_names(&self) -> Vec<&str> {
        todo!()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        todo!()
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        todo!()
    }

    fn properties_names(&self) -> Vec<&str> {
        todo!()
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        todo!()
    }

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        todo!()
    }
}
// [impl]
