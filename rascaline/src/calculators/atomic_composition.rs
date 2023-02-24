use std::sync::Arc;

use equistore::{Labels, LabelsBuilder, TensorMap};

use crate::{Error, System};

use super::CalculatorBase;
use crate::labels::{CenterSpeciesKeys, KeysBuilder};


/// An atomic composition calculator for obtaining the stoichiometric information.
///
/// For `per_structure=False` calculator has one property `count` that is
/// `1` for all centers, and has a sample index that indicates the central atom type.
///
/// For `per_structure=True` a sum for each structure is performed and the number of
/// atoms per structure is saved. The only sample left is names ``structure``.
///
/// Positions/cell gradients of the composition are zero everywhere. Therefore, the
/// gradient data will only be an empty array.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct AtomicComposition {
    /// Sum atom numbers for each structure.
    pub per_structure: bool,
}

impl CalculatorBase for AtomicComposition {
    fn name(&self) -> String {
        return "atom-centered composition features".into();
    }

    fn parameters(&self) -> String {
        return serde_json::to_string(self).expect("failed to serialize to JSON");
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        return CenterSpeciesKeys.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        if self.per_structure {
            return vec!["structure"];
        }

        return vec!["structure", "center"];
    }

    fn samples(
        &self,
        keys: &Labels,
        systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center"]);
        let mut samples = Vec::new();
        for [species_center_key] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(self.samples_names());

            for (system_i, system) in systems.iter_mut().enumerate() {
                if self.per_structure {
                    builder.add(&[system_i]);
                } else {
                    let species = system.species()?;

                    for (center_i, &species_center_sys) in species.iter().enumerate() {
                        if species_center_key.i32() == species_center_sys {
                            builder.add(&[system_i, center_i]);
                        }
                    }
                }
            }
            samples.push(Arc::new(builder.finish()));
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
        _samples: &[Arc<Labels>],
        _systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Arc<Labels>>, Error> {
        // Positions/cell gradients of the composition are zero everywhere.
        // Therefore, we only return a vector of empty labels (one for each key).
        let gradient_samples = Arc::new(Labels::empty(vec!["sample", "structure", "atom"]));
        return Ok(vec![gradient_samples; keys.count()]);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Arc<Labels>>> {
        return vec![Vec::new(); keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        return vec!["count"];
    }

    fn properties(&self, keys: &Labels) -> Vec<Arc<Labels>> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        properties.add(&[0]);
        let properties = Arc::new(properties.finish());

        return vec![properties; keys.count()];
    }

    fn compute(
        &mut self,
        systems: &mut [Box<dyn System>],
        descriptor: &mut TensorMap,
    ) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["species_center"]);

        for (key, mut block) in descriptor.iter_mut() {
            let species_center = key[0].i32();

            let values = block.values_mut();
            let array = values.data.as_array_mut();

            for (property_i, &[count]) in values.properties.iter_fixed_size().enumerate() {
                if count == 0 {
                    for (sample_i, samples) in values.samples.iter().enumerate() {
                        let mut value = 0.0;

                        if self.per_structure {
                            // Current system is saved in the 0th index of the samples.
                            let system_i = samples[0].usize();
                            let system = &systems[system_i];
                            for &species in system.species()? {
                                if species == species_center {
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
    use equistore::Labels;
    use ndarray::array;

    use crate::systems::test_utils::{test_system, test_systems};
    use crate::Calculator;

    use super::super::CalculatorBase;
    use super::AtomicComposition;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_structure: false,
        }) as Box<dyn CalculatorBase>);

        assert_eq!(calculator.name(), "atom-centered composition features");
        assert_eq!(calculator.parameters(), "{\"per_structure\":false}");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(AtomicComposition {
            per_structure: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator
            .compute(&mut systems, Default::default())
            .unwrap();

        assert_eq!(descriptor.blocks().len(), 2);
        // test against hydrogen block which has the id `1`
        let values = descriptor.block_by_id(1).values().data.as_array();
        assert_eq!(values.shape(), [2, 1]);

        assert_eq!(values, array![[1.0], [1.0]].into_dyn());
    }

    #[test]
    fn values_per_structure() {
        let mut calculator = Calculator::from(Box::new(AtomicComposition {
            per_structure: true,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator
            .compute(&mut systems, Default::default())
            .unwrap();

        assert_eq!(descriptor.blocks().len(), 2);
        //test against hydrogen block which has the id `1`
        let values = descriptor.block_by_id(1).values().data.as_array();
        assert_eq!(values.shape(), [1, 1]);

        assert_eq!(values, array![[2.0]].into_dyn());
    }

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_structure: false,
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
    fn finite_differences_positions_per_structure() {
        let calculator = Calculator::from(Box::new(AtomicComposition {
            per_structure: true,
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
            per_structure: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let keys = Labels::new(["species_center"], &[[1], [6], [8], [-42]]);
        let samples = Labels::new(["structure", "center"], &[[0, 1]]);
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
