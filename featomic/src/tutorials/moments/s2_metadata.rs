use metatensor::{Labels, TensorMap, LabelsBuilder};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsTypesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, AtomicTypeFilter};
use crate::calculators::CalculatorBase;

// [struct]
#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
}
// [struct]

impl CalculatorBase for GeometricMoments {
    // [CalculatorBase::name]
    fn name(&self) -> String {
        "geometric moments".to_string()
    }
    // [CalculatorBase::name]

    // [CalculatorBase::parameters]
    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }
    // [CalculatorBase::parameters]

    // [CalculatorBase::cutoffs]
    fn cutoffs(&self) -> &[f64] {
        std::slice::from_ref(&self.cutoff)
    }
    // [CalculatorBase::cutoffs]

    // [CalculatorBase::keys]
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let builder = CenterSingleNeighborsTypesKeys {
            cutoff: self.cutoff,
            // self pairs would have a distance of 0 and would not contribute
            // anything meaningful to a GeometricMoments representation
            self_pairs: false,
        };
        return builder.keys(systems);
    }
    // [CalculatorBase::keys]

    // [CalculatorBase::samples]
    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_type"]);

        let mut samples = Vec::new();
        for [center_type, neighbor_type] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                // only include central atoms of this type
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                // with a neighbor of this type somewhere in the neighborhood
                // defined by the spherical `cutoff`.
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: false,
            };

            samples.push(builder.samples(systems)?);
        }

        return Ok(samples);
    }
    // [CalculatorBase::samples]

    // [CalculatorBase::supports_gradient]
    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            _ => false,
        }
    }
    // [CalculatorBase::supports_gradient]

    // [CalculatorBase::positions_gradient_samples]
    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type", "neighbor_type"]);
        debug_assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([center_type, neighbor_type], samples_for_key) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                // only include gradients with respect to neighbor atoms with
                // this atomic type (the other atoms do not contribute to the
                // gradients in the current block).
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: false,
            };

            gradient_samples.push(builder.gradients_for(systems, samples_for_key)?);
        }

        return Ok(gradient_samples);
    }
    // [CalculatorBase::positions_gradient_samples]

    // [CalculatorBase::components]
    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        return vec![vec![]; keys.count()];
    }
    // [CalculatorBase::components]

    // [CalculatorBase::properties]
    fn property_names(&self) -> Vec<&str> {
        vec!["k"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut builder = LabelsBuilder::new(self.property_names());
        for k in 0..=self.max_moment {
            builder.add(&[k]);
        }
        let properties = builder.finish();

        return vec![properties; keys.count()];
    }
    // [CalculatorBase::properties]

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        todo!()
    }

}
