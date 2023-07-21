use log::{info, warn};

use equistore::TensorMap;
use equistore::{Labels, LabelsBuilder};

use super::CalculatorBase;
use crate::labels::{SpeciesFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{CenterSpeciesKeys, KeysBuilder};

use crate::{Error, System};

/// A stupid calculator implementation used to test the API, and API binding to
/// C/Python/etc.
///
/// The calculator has two features: one containing the atom index +
/// `self.delta`, and the other one containing `x + y + z`.
#[doc(hidden)]
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct DummyCalculator {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Delta added to the atom index in the first feature
    pub delta: isize,
    /// Unused name parameter, to test passing string values
    pub name: String,
}

impl CalculatorBase for DummyCalculator {
    fn name(&self) -> String {
        // abusing the name as description
        format!("dummy test calculator with cutoff: {} - delta: {} - name: {}",
            self.cutoff, self.delta, self.name
        )
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        std::slice::from_ref(&self.cutoff)
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        return CenterSpeciesKeys.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["species_center"]);
        let mut samples = Vec::new();
        for [species_center] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Any,
                self_pairs: false,
            };

            samples.push(builder.samples(systems)?);
        }

        return Ok(samples);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        debug_assert_eq!(keys.count(), samples.len());
        let mut gradient_samples = Vec::new();
        for ([species_center], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples{
                cutoff: self.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Any,
                self_pairs: false,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        return vec![Vec::new(); keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["index_delta", "x_y_z"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        properties.add(&[1, 0]);
        properties.add(&[0, 1]);
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "DummyCalculator::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        if self.name.contains("log-test-info:") {
            info!("{}", self.name);
        } else if self.name.contains("log-test-warn:") {
            warn!("{}", self.name);
        }

        for (key, mut block) in descriptor.iter_mut() {
            let species_center = key[0].i32();

            let block_data = block.data_mut();
            let array = block_data.values.to_array_mut();

            for (sample_i, [system, center]) in block_data.samples.iter_fixed_size().enumerate() {
                let system_i = system.usize();
                let center_i = center.usize();

                debug_assert_eq!(systems[system_i].species()?[center_i], species_center);

                for (property_i, property) in block_data.properties.iter().enumerate() {
                    if property[0].i32() == 1 {
                        array[[sample_i, property_i]] = center_i as f64 + self.delta as f64;
                    } else if property[1].i32() == 1 {
                        let system = &mut *systems[system_i];
                        system.compute_neighbors(self.cutoff)?;

                        let positions = system.positions()?;
                        let mut sum = positions[center_i][0] + positions[center_i][1] + positions[center_i][2];
                        for pair in system.pairs()? {
                            if pair.first == center_i {
                                sum += positions[pair.second][0] + positions[pair.second][1] + positions[pair.second][2];
                            }

                            if pair.second == center_i {
                                sum += positions[pair.first][0] + positions[pair.first][1] + positions[pair.first][2];
                            }
                        }

                        array[[sample_i, property_i]] = sum;
                    }
                }
            }

            if let Some(mut gradient) = block.gradient_mut("positions") {
                let gradient = gradient.data_mut();
                let array = gradient.values.to_array_mut();

                for gradient_sample_i in 0..array.shape()[0] {
                    for (property_i, property) in gradient.properties.iter().enumerate() {
                        if property[0].i32() == 1 {
                            array[[gradient_sample_i, 0, property_i]] = 0.0;
                            array[[gradient_sample_i, 1, property_i]] = 0.0;
                            array[[gradient_sample_i, 2, property_i]] = 0.0;
                        } else if property[1].i32() == 1 {
                            array[[gradient_sample_i, 0, property_i]] = 1.0;
                            array[[gradient_sample_i, 1, property_i]] = 1.0;
                            array[[gradient_sample_i, 2, property_i]] = 1.0;
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{s, aview1};
    use equistore::Labels;

    use crate::systems::test_utils::test_systems;
    use crate::Calculator;

    use super::DummyCalculator;
    use super::super::CalculatorBase;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.4,
            delta: 9,
            name: "a long name".into(),
        }) as Box<dyn CalculatorBase>);

        assert_eq!(
            calculator.name(),
            "dummy test calculator with cutoff: 1.4 - delta: 9 - name: a long name"
        );

        assert_eq!(
            calculator.parameters(),
            "{\"cutoff\":1.4,\"delta\":9,\"name\":\"a long name\"}"
        );
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.0,
            delta: 9,
            name: String::new(),
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        let keys = descriptor.keys();
        assert_eq!(keys.names(), ["species_center"]);
        assert_eq!(keys.count(), 2);
        assert_eq!(keys[0], [-42]);
        assert_eq!(keys[1], [1]);

        let o_block = &descriptor.block_by_id(0);
        let values = o_block.values().to_array();
        assert_eq!(values.shape(), [1, 2]);
        assert_eq!(values.slice(s![0, ..]), aview1(&[9.0, -1.1778999999999997]));

        let h_block = &descriptor.block_by_id(1);
        let values = h_block.values().to_array();
        assert_eq!(values.shape(), [2, 2]);
        assert_eq!(values.slice(s![0, ..]), aview1(&[10.0, 0.16649999999999998]));
        assert_eq!(values.slice(s![1, ..]), aview1(&[11.0, -1.3443999999999998]));
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.0,
            delta: 9,
            name: String::new(),
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water"]);

        let samples = Labels::new(["structure", "center"], &[[0, 1]]);
        let properties = Labels::new(["index_delta", "x_y_z"], &[[0, 1]]);
        let keys = Labels::new(["species_center"], &[[0], [1], [6], [-42]]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }
}
