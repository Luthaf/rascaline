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

// variables declared as const to make the code compile
const first_sample_position: Option<usize> = None;
const second_sample_position: Option<usize> = None;

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
            // ...

            for pair in system.pairs()? {
                // ...

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;

                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    let k = feature[0].usize();
                    let moment = f64::powi(pair.distance, k as i32);

                    if let Some(i_first_sample) = first_sample_position {
                        descriptor.values[[i_first_sample, i_feature]] += moment / n_neighbors_first;
                    }

                    if let Some(i_second_sample) = second_sample_position {
                        descriptor.values[[i_second_sample, i_feature]] += moment / n_neighbors_second;
                    }
                }

            }
        }

        return Ok(());
    }
    // [compute]
}
