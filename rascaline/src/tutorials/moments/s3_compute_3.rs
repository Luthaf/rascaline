use crate::{System, Descriptor, Error};
use crate::descriptor::{SamplesBuilder, TwoBodiesSpeciesSamples};
use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
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

    fn get_parameters(&self) -> String {
        todo!()
    }

    fn compute_gradients(&self) -> bool {
        todo!()
    }

    fn features_names(&self) -> Vec<&str> {
        todo!()
    }

    fn features(&self) -> Indexes {
        todo!()
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        todo!()
    }

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        todo!()
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        // ...

        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;

            // add this line
            let species = system.species()?;

            for pair in system.pairs()? {
                // sample with the first atom in the pair as center
                let first_sample = [
                    IndexValue::from(i_system),             // system
                    IndexValue::from(pair.first),           // center
                    IndexValue::from(species[pair.first]),  // species_center
                    IndexValue::from(species[pair.second]), // species_neighbor
                ];

                // sample with the second atom in the pair as center
                let second_sample = [
                    IndexValue::from(i_system),
                    IndexValue::from(pair.second),
                    IndexValue::from(species[pair.second]),
                    IndexValue::from(species[pair.first]),
                ];

                // get the positions of the samples. These variables will be
                // `None` if the samples are not present in the descriptor, i.e.
                // if the user did not request them.
                let first_sample_position = descriptor.samples.position(&first_sample);
                let second_sample_position = descriptor.samples.position(&second_sample);

                // skip calculation if neither of the samples is present
                if first_sample_position.is_none() && second_sample_position.is_none() {
                    continue;
                }

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;


            }
        }

        return Ok(());
    }
    // [compute]
}
