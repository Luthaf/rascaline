use crate::{System, Descriptor, Error};
use crate::descriptor::{SamplesBuilder, TwoBodiesSpeciesSamples};
use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
use crate::calculators::CalculatorBase;

// [struct]
#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
    gradients: bool,
}
// [struct]

impl CalculatorBase for GeometricMoments {
    // [CalculatorBase::name]
    fn name(&self) -> String {
        "geometric moments".to_string()
    }
    // [CalculatorBase::name]

    // [CalculatorBase::get_parameters]
    fn get_parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }
    // [CalculatorBase::get_parameters]

    // [CalculatorBase::compute_gradients]
    fn compute_gradients(&self) -> bool {
        self.gradients
    }
    // [CalculatorBase::compute_gradients]

    // [CalculatorBase::features_names]
    fn features_names(&self) -> Vec<&str> {
        vec!["k"]
    }
    // [CalculatorBase::features_names]

    // [CalculatorBase::features]
    fn features(&self) -> Indexes {
        let mut builder = IndexesBuilder::new(self.features_names());
        for k in 0..=self.max_moment {
            builder.add(&[IndexValue::from(k)]);
        }

        return builder.finish();
    }
    // [CalculatorBase::features]

    // [CalculatorBase::check_features]
    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        assert_eq!(indexes.names(), self.features_names());

        for value in indexes {
            if value[0].usize() > self.max_moment {
                return Err(Error::InvalidParameter(format!(
                    "'k' is too large for this GeometricMoments calculator: \
                    expected value below {}, got {}", self.max_moment, value[0]
                )));
            }
        }

        return Ok(());
    }
    // [CalculatorBase::check_features]

    // [CalculatorBase::samples_builder]
    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        Box::new(TwoBodiesSpeciesSamples::new(self.cutoff))
    }
    // [CalculatorBase::samples_builder]

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        todo!()
    }
}
