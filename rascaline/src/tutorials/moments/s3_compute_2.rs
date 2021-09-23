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

            for pair in system.pairs()? {
                // more code to come here
            }
        }

        return Ok(());
    }
    // [compute]
}
