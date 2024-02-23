use log::{info, warn};

use metatensor::TensorMap;
use metatensor::{Labels, LabelsBuilder};

use super::CalculatorBase;
use crate::labels::{AtomicTypeFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{CenterTypesKeys, KeysBuilder};

use crate::{Error, System, Vector3D};

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
        return CenterTypesKeys.keys(systems);
    }

    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type"]);
        let mut samples = Vec::new();
        for [center_type] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Any,
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
        for ([center_type], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples{
                cutoff: self.cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Any,
                self_pairs: false,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        return vec![Vec::new(); keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["index_delta", "x_y_z"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
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

        for (key, mut block) in descriptor {
            let center_type = key[0].i32();

            let block_data = block.data_mut();
            let array = block_data.values.to_array_mut();

            for (sample_i, [system, atom]) in block_data.samples.iter_fixed_size().enumerate() {
                let system_i = system.usize();
                let atom_i = atom.usize();

                debug_assert_eq!(systems[system_i].types()?[atom_i], center_type);

                for (property_i, property) in block_data.properties.iter().enumerate() {
                    if property[0].i32() == 1 {
                        array[[sample_i, property_i]] = atom_i as f64 + self.delta as f64;
                    } else if property[1].i32() == 1 {
                        let system = &mut *systems[system_i];
                        system.compute_neighbors(self.cutoff)?;

                        let positions = system.positions()?;
                        let cell = system.cell()?.matrix();
                        let mut sum = positions[atom_i][0] + positions[atom_i][1] + positions[atom_i][2];
                        for pair in system.pairs()? {
                            // this code just check for consistency in the
                            // neighbor list
                            let shift = pair.cell_shift_indices[0] as f64 * Vector3D::from(cell[0])
                                + pair.cell_shift_indices[1] as f64 * Vector3D::from(cell[1])
                                + pair.cell_shift_indices[2] as f64 * Vector3D::from(cell[2]);
                            let from_shift = positions[pair.second] - positions[pair.first] + shift;
                            if !approx::relative_eq!(from_shift, pair.vector, max_relative=1e-6) {
                                return Err(Error::InvalidParameter(format!(
                                    "system implementation returned inconsistent neighbors list:\
                                    pair.vector is {:?}, but the cell shift give {:?} for atoms {}-{}",
                                    pair.vector, from_shift, pair.first, pair.second
                                )));
                            }

                            if !approx::relative_eq!(pair.vector.norm(), pair.distance, max_relative=1e-6) {
                                return Err(Error::InvalidParameter(format!(
                                    "system implementation returned inconsistent neighbors list:\
                                    pair.vector norm is {}, but pair.distance is {} for atoms {}-{}",
                                    pair.vector.norm(), pair.distance, pair.first, pair.second
                                )));
                            }

                            let pairs_by_center = system.pairs_containing(pair.first)?;
                            if !pairs_by_center.iter().any(|p| p == pair) {
                                return Err(Error::InvalidParameter(format!(
                                    "system implementation returned inconsistent neighbors list:\
                                    pairs_containing({}) does not contains a pair for atoms {}-{}",
                                    pair.first, pair.first, pair.second
                                )));
                            }

                            let pairs_by_center = system.pairs_containing(pair.second)?;
                            if !pairs_by_center.iter().any(|p| p == pair) {
                                return Err(Error::InvalidParameter(format!(
                                    "system implementation returned inconsistent neighbors list:\
                                    pairs_containing({}) does not contains a pair for atoms {}-{}",
                                    pair.second, pair.first, pair.second
                                )));
                            }
                            // end of neighbors list consistency check

                            // actual values calculation
                            if pair.first == atom_i {
                                sum += positions[pair.second][0] + positions[pair.second][1] + positions[pair.second][2];
                            }

                            if pair.second == atom_i {
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
    use metatensor::Labels;

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
        assert_eq!(keys.names(), ["center_type"]);
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

        let samples = Labels::new(["system", "atom"], &[[0, 1]]);
        let properties = Labels::new(["index_delta", "x_y_z"], &[[0, 1]]);
        let keys = Labels::new(["center_type"], &[[0], [1], [6], [-42]]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }
}
