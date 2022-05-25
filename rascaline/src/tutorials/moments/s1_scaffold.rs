// [imports]
use std::sync::Arc;

use equistore::{Labels, TensorMap, LabelsBuilder, LabelValue};

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
    gradients: bool,
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

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        todo!()
    }

    fn gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Option<Vec<Arc<Labels>>>, Error> {
        todo!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Arc<Labels>>> {
        todo!()
    }

    fn properties_names(&self) -> Vec<&str> {
        todo!()
    }

    fn properties(&self, keys: &Labels) -> Vec<Arc<Labels>> {
        todo!()
    }

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        todo!()
    }
}
// [impl]
