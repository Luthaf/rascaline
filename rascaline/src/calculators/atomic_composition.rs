use std::sync::Arc;

use equistore::{Labels, LabelsBuilder, TensorMap};

use crate::{Error, System};

use super::CalculatorBase;
use crate::labels::{CenterSpeciesKeys, KeysBuilder};
use crate::labels::{SamplesBuilder, SpeciesFilter};
use crate::labels::{SamplesPerAtom, Structures};


/// An atomic composition calculator for obtaining the stoichiometric information.
///
/// For `per_structure=False` calculator has one property `count` that is
/// `1` for all centers, and has a sample index that indicates the central atom type.
///
/// For `per_structure=True` a sum for each structure is performed and the number of 
/// atoms per structure is saved. The only sample left is names ``structure``.
pub struct AtomicComposition {
    // Define if the atom numbers should be summed for each structure.
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
        for [species_center] in keys.iter_fixed_size() {
            if self.per_structure {
                let builder = Structures {
                    species_center: SpeciesFilter::Single(species_center.i32()),
                };
                samples.push(builder.samples(systems)?);
            } else {
                let builder = SamplesPerAtom {
                    species_center: SpeciesFilter::Single(species_center.i32()),
                };
                samples.push(builder.samples(systems)?);
            };
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
        samples: &[Arc<Labels>],
        systems: &mut [Box<dyn System>],
    ) -> Result<Vec<Arc<Labels>>, Error> {
        debug_assert_eq!(keys.count(), samples.len());
        let mut gradient_samples = Vec::new();
        for ([species_center], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = SamplesPerAtom {
                species_center: SpeciesFilter::Single(species_center.i32()),
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
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
            let values = block.values_mut();
            let array = values.data.as_array_mut();

            if self.per_structure {
                let species = key[0].i32();
                for (system_i, system) in systems.iter_mut().enumerate() {
                    let all_species = system.species()?;
                    let mut n_species = 0.;

                    for &s in all_species {
                        if s == species {
                            n_species += 1.;
                        }
                    }
                    array[[system_i, 0]] = n_species;
                }
            } else {
                array.fill(1.0);
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
