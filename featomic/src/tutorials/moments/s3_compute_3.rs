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
        assert_eq!(descriptor.keys().names(), ["center_type", "neighbor_type"]);

        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;

            // add this line
            let types = system.types()?;

            for pair in system.pairs()? {
                // get the block where the first atom is the center
                let first_block_id = descriptor.keys().position(&[
                    types[pair.first].into(), types[pair.second].into(),
                ]);

                // get the sample corresponding to the first atom as a center
                //
                // This will be `None` if the block or samples are not present
                // in the descriptor, i.e. if the user did not request them.
                let first_sample_position = if let Some(block_id) = first_block_id {
                    descriptor.block_by_id(block_id).samples().position(&[
                        system_i.into(), pair.first.into()
                    ])
                } else {
                    None
                };

                // get the id of the block where the second atom is the center
                let second_block_id = descriptor.keys().position(&[
                    types[pair.second].into(), types[pair.first].into(),
                ]);
                // get the sample corresponding to the first atom as a center
                let second_sample_position = if let Some(block_id) = second_block_id {
                    descriptor.block_by_id(block_id).samples().position(&[
                        system_i.into(), pair.second.into()
                    ])
                } else {
                    None
                };

                // skip calculation if neither of the samples is present
                if first_sample_position.is_none() && second_sample_position.is_none() {
                    continue;
                }

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;

                // more code coming up here!
            }
        }

        return Ok(());
    }
    // [compute]
}
