use metatensor::{Labels, TensorMap, LabelsBuilder};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsTypesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, AtomicTypeFilter};
use crate::calculators::CalculatorBase;

#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
}

impl CalculatorBase for GeometricMoments {
    fn name(&self) -> String {
        todo!()
    }

    fn parameters(&self) -> String {
        todo!()
    }

    fn cutoffs(&self) -> &[f64] {
        todo!()
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        todo!()
    }

    fn sample_names(&self) -> Vec<&str> {
        todo!()
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        todo!()
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        todo!()
    }

    fn property_names(&self) -> Vec<&str> {
        todo!()
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        todo!()
    }

    // [compute]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["center_type", "neighbor_type"]);

        // we'll add more code here

        return Ok(());
    }
    // [compute]
}
