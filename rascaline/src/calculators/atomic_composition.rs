use metatensor::{Labels, LabelsBuilder, TensorMap};

use crate::{Error, System};

use super::CalculatorBase;
use crate::labels::{CenterTypesKeys, KeysBuilder};


/// An atomic composition calculator for obtaining the stoichiometric
/// information.
///
/// For `per_system=False` calculator has one property `count` that is `1` for
/// all atoms, and has a sample index that indicates the central atom type.
///
/// For `per_system=True` a sum for each system is performed and the number of
/// atoms per system is saved. The only sample left is names `system`.
///
/// Positions/cell gradients of the composition are zero everywhere. Therefore,
/// the gradient data will only be an empty array.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct AtomicComposition {
    /// Sum atom numbers for each system.
    pub per_system: bool,
}

impl CalculatorBase for AtomicComposition {
    fn name(&self) -> String {
        return "atom-centered composition features".into();
    }

    fn parameters(&self) -> String {
        return serde_json::to_string(self).expect("failed to serialize to JSON");
    }

    fn cutoffs(&self) -> &[f64] {
        &[]
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        return CenterTypesKeys.keys(systems);
    }

    fn sample_names(&self) -> Vec<&str> {
        if self.per_system {
            return vec!["system"];
        }

        return vec!["system", "atom"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["center_type"]);
        let mut samples = Vec::new();
        for [center_type_key] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(self.sample_names());

            for (system_i, system) in systems.iter_mut().enumerate() {
                if self.per_system {
                    builder.add(&[system_i]);
                } else {
                    let types = system.types()?;

                    for (center_i, &center_type) in types.iter().enumerate() {
                        if center_type_key.i32() == center_type {
                            builder.add(&[system_i, center_i]);
                        }
                    }
                }
            }
            samples.push(builder.finish());
        }

        return Ok(samples);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            "cell" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(
        &self,
        keys: &Labels,
        _samples: &[Labels],
        _systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Labels>, Error> {
        // Positions/cell gradients of the composition are zero everywhere.
        // Therefore, we only return a vector of empty labels (one for each key).
        let gradient_samples = Labels::empty(vec!["sample", "system", "atom"]);
        return Ok(vec![gradient_samples; keys.count()]);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        return vec![Vec::new(); keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        return vec!["count"];
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        properties.add(&[0]);
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    fn compute(
        &mut self,
        systems: &mut [Box<dyn System>],
        descriptor: &mut TensorMap,
    ) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["center_type"]);

        for (key, mut block) in descriptor {
            let center_type = key[0].i32();

            let block = block.data_mut();
            let array = block.values.to_array_mut();

            for (property_i, &[count]) in block.properties.iter_fixed_size().enumerate() {
                if count == 0 {
                    for (sample_i, samples) in block.samples.iter().enumerate() {
                        let mut value = 0.0;

                        if self.per_system {
                            // Current system is saved in the 0th index of the samples.
                            let system_i = samples[0].usize();
                            let system = &systems[system_i];
                            for &atomic_type in system.types()? {
                                if atomic_type == center_type {
                                    value += 1.0;
                                }
                            }
                        } else {
                            value += 1.0;
                        }
                        array[[sample_i, property_i]] = value;
                    }
                }
            }
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use metatensor::Labels;
    use ndarray::array;

    use crate::systems::test_utils::{test_system, test_systems};
    use crate::Calculator;

    use super::super::CalculatorBase;
    use super::AtomicComposition;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: false,
        }) as Box<dyn CalculatorBase>);

        assert_eq!(calculator.name(), "atom-centered composition features");
        assert_eq!(calculator.parameters(), "{\"per_system\":false}");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator
            .compute(&mut systems, Default::default())
            .unwrap();

        assert_eq!(descriptor.blocks().len(), 2);
        // test against hydrogen block which has the id `1`
        let values = descriptor.block_by_id(1).values().to_array();
        assert_eq!(values.shape(), [2, 1]);

        assert_eq!(values, array![[1.0], [1.0]].into_dyn());
    }

    #[test]
    fn values_per_system() {
        let mut calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: true,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator
            .compute(&mut systems, Default::default())
            .unwrap();

        assert_eq!(descriptor.blocks().len(), 2);
        // test against hydrogen block which has the id `1`
        let values = descriptor.block_by_id(1).values().to_array();
        assert_eq!(values.shape(), [1, 1]);

        assert_eq!(values, array![[2.0]].into_dyn());
    }

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: false,
        }) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn finite_differences_positions_per_system() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: true,
        }) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_system: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let keys = Labels::new(["center_type"], &[[1], [6], [8], [-42]]);
        let samples = Labels::new(["system", "atom"], &[[0, 1]]);
        let properties = Labels::new(["count"], &[[0]]);

        crate::calculators::tests_utils::compute_partial(
            calculator,
            &mut systems,
            &keys,
            &samples,
            &properties,
        );
    }
}
