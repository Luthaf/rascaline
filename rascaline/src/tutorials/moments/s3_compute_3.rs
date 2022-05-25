use std::sync::Arc;

use equistore::{Labels, TensorMap, LabelsBuilder, LabelValue};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsSpeciesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, SpeciesFilter};
use crate::calculators::CalculatorBase;

#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
    gradients: bool,
}

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

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor"]);

        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;

            // add this line
            let species = system.species()?;

            for pair in system.pairs()? {
                // get the block where the first atom is the center
                let first_block_id = descriptor.keys().position(&[
                    species[pair.first].into(), species[pair.second].into(),
                ]).expect("missing block for the first atom");
                let first_block = &descriptor.blocks()[first_block_id];

                // get the id of the block where the second atom is the center
                let second_block_id = descriptor.keys().position(&[
                    species[pair.second].into(), species[pair.first].into(),
                ]).expect("missing block for the second atom");
                let second_block = &descriptor.blocks()[second_block_id];

                // get the positions of the samples in their respective blocks.
                // These variables will be `None` if the samples are not present
                // in the blocks, i.e. if the user did not request them.
                let first_sample_position = first_block.values().samples.position(&[
                    system_i.into(), pair.first.into()
                ]);
                let second_sample_position = second_block.values().samples.position(&[
                    system_i.into(), pair.second.into()
                ]);

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
