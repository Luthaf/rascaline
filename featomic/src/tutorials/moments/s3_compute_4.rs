
use metatensor::{Labels, TensorMap, LabelsBuilder};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsTypesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, AtomicTypeFilter};
use crate::calculators::CalculatorBase;

// these are here just to make the code below compile
const first_sample_position: Option<usize> = None;
const second_sample_position: Option<usize> = None;
const first_block_id: Option<usize> = None;
const second_block_id: Option<usize> = None;

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

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        todo!()
    }

    fn sample_names(&self) -> Vec<&str> {
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

    fn property_names(&self) -> Vec<&str> {
        todo!()
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        todo!()
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        // ...
        for (system_i, system) in systems.iter_mut().enumerate() {
            // ...
            for pair in system.pairs()? {
                // ...

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;

                if let Some(sample_i) = first_sample_position {
                    let block_id = first_block_id.expect("we have a sample in this block");
                    let mut block = descriptor.block_mut_by_id(block_id);
                    let block = block.data_mut();
                    let array = block.values.to_array_mut();

                    for (property_i, [k]) in block.properties.iter_fixed_size().enumerate() {
                        let value = f64::powi(pair.distance, k.i32()) / n_neighbors_first;
                        array[[sample_i, property_i]] += value;
                    }
                }

                if let Some(sample_i) = second_sample_position {
                    let block_id = second_block_id.expect("we have a sample in this block");
                    let mut block = descriptor.block_mut_by_id(block_id);
                    let block = block.data_mut();
                    let array = block.values.to_array_mut();

                    for (property_i, [k]) in block.properties.iter_fixed_size().enumerate() {
                        let value = f64::powi(pair.distance, k.i32()) / n_neighbors_second;
                        array[[sample_i, property_i]] += value;
                    }
                }

                // more code coming up
            }
        }
        return Ok(());
    }
    // [compute]
}
