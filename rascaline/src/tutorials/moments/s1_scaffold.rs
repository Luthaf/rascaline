// [imports]
use crate::{System, Descriptor, Error};
use crate::descriptor::{SamplesBuilder, TwoBodiesSpeciesSamples};
use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
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

    fn get_parameters(&self) -> String {
        todo!()
    }

    fn features_names(&self) -> Vec<&str> {
        todo!()
    }

    fn features(&self) -> Indexes {
        todo!()
    }

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        todo!()
    }

    fn compute_gradients(&self) -> bool {
        todo!()
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        todo!()
    }

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        todo!()
    }
}
// [impl]
