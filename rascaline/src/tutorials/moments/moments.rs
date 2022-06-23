use std::sync::Arc;

use equistore::{Labels, TensorMap, LabelsBuilder, LabelValue};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsSpeciesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, SpeciesFilter};
use crate::calculators::CalculatorBase;

#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
}

impl CalculatorBase for GeometricMoments {
    fn name(&self) -> String {
        "geometric moments".to_string()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let builder = CenterSingleNeighborsSpeciesKeys {
            cutoff: self.cutoff,
            self_pairs: false,
        };
        return builder.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor"]);

        let mut samples = Vec::new();
        for [species_center, species_neighbor] in keys.iter_fixed_size() {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
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

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["species_center", "species_neighbor"]);
        debug_assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([center_species, species_neighbor], samples_for_key) in keys.iter_fixed_size().zip(samples) {
            let builder = AtomCenteredSamples {
                cutoff: self.cutoff,
                species_center: SpeciesFilter::Single(center_species.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                self_pairs: false,
            };

            gradient_samples.push(builder.gradients_for(systems, samples_for_key)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Arc<Labels>>> {
        return vec![vec![]; keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["k"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Arc<Labels>> {
        let mut builder = LabelsBuilder::new(self.properties_names());
        for k in 0..=self.max_moment {
            builder.add(&[k]);
        }
        let properties = Arc::new(builder.finish());

        return vec![properties; keys.count()];
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor"]);

        let do_positions_gradients = descriptor.blocks()[0].gradient("positions").is_some();

        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            for pair in system.pairs()? {
                let first_block_id = descriptor.keys().position(&[
                    species[pair.first].into(), species[pair.second].into(),
                ]).expect("missing block for the first atom");
                let first_block = &descriptor.blocks()[first_block_id];

                let second_block_id = descriptor.keys().position(&[
                    species[pair.second].into(), species[pair.first].into(),
                ]).expect("missing block for the second atom");
                let second_block = &descriptor.blocks()[second_block_id];


                let first_sample_position = first_block.values().samples.position(&[
                    system_i.into(), pair.first.into()
                ]);
                let second_sample_position = second_block.values().samples.position(&[
                    system_i.into(), pair.second.into()
                ]);

                if first_sample_position.is_none() && second_sample_position.is_none() {
                    continue;
                }

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;

                if let Some(sample_i) = first_sample_position {
                    let mut block = descriptor.block_mut(first_block_id);
                    let values = block.values_mut();
                    let array = values.data.as_array_mut();

                    for (property_i, [k]) in values.properties.iter_fixed_size().enumerate() {
                        let value = f64::powi(pair.distance, k.i32()) / n_neighbors_first;
                        array[[sample_i, property_i]] += value;
                    }
                }

                if let Some(sample_i) = second_sample_position {
                    let mut block = descriptor.block_mut(second_block_id);
                    let values = block.values_mut();
                    let array = values.data.as_array_mut();

                    for (property_i, [k]) in values.properties.iter_fixed_size().enumerate() {
                        let value = f64::powi(pair.distance, k.i32()) / n_neighbors_second;
                        array[[sample_i, property_i]] += value;
                    }
                }

                if do_positions_gradients {
                    let mut moment_gradients = Vec::new();
                    for k in 0..=self.max_moment {
                        moment_gradients.push([
                            pair.vector[0] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                            pair.vector[1] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                            pair.vector[2] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                        ]);
                    }

                    if let Some(sample_position) = first_sample_position {
                        let mut block = descriptor.block_mut(first_block_id);
                        let gradient = block.gradient_mut("positions").expect("missing gradient storage");
                        let array = gradient.data.as_array_mut();

                        let gradient_wrt_second = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.second.into()
                        ]);
                        let gradient_wrt_self = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.first.into()
                        ]);

                        for (property_i, [k]) in gradient.properties.iter_fixed_size().enumerate() {
                            if let Some(sample_i) = gradient_wrt_second {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] += grad[0] / n_neighbors_first;
                                array[[sample_i, 1, property_i]] += grad[1] / n_neighbors_first;
                                array[[sample_i, 2, property_i]] += grad[2] / n_neighbors_first;
                            }

                            if let Some(sample_i) = gradient_wrt_self {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] -= grad[0] / n_neighbors_first;
                                array[[sample_i, 1, property_i]] -= grad[1] / n_neighbors_first;
                                array[[sample_i, 2, property_i]] -= grad[2] / n_neighbors_first;
                            }
                        }
                    }

                    if let Some(sample_position) = second_sample_position {
                        let mut block = descriptor.block_mut(second_block_id);
                        let gradient = block.gradient_mut("positions").expect("missing gradient storage");
                        let array = gradient.data.as_array_mut();

                        let gradient_wrt_first = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.first.into()
                        ]);
                        let gradient_wrt_self = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.second.into()
                        ]);

                        for (property_i, [k]) in gradient.properties.iter_fixed_size().enumerate() {
                            if let Some(sample_i) = gradient_wrt_first {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] -= grad[0] / n_neighbors_second;
                                array[[sample_i, 1, property_i]] -= grad[1] / n_neighbors_second;
                                array[[sample_i, 2, property_i]] -= grad[2] / n_neighbors_second;
                            }

                            if let Some(sample_i) = gradient_wrt_self {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] += grad[0] / n_neighbors_second;
                                array[[sample_i, 1, property_i]] += grad[1] / n_neighbors_second;
                                array[[sample_i, 2, property_i]] += grad[2] / n_neighbors_second;
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }
    // [compute]
}


#[allow(clippy::eq_op)]
// [property-test]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Calculator;
    use crate::systems::test_utils::test_systems;

    use ndarray::array;

    #[test]
    fn zeroth_moment() {
        // Create a Calculator wrapping a GeometricMoments instance
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 0,
        }) as Box<dyn CalculatorBase>);

        // create a bunch of systems in a format compatible with `calculator.compute`.
        // Available systems include "water" and "methane" for the corresponding
        // molecules, and "CH" for a basic 2 atoms system.
        let mut systems = test_systems(&["water", "CH"]);

        // run the calculation using default parameters
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        // check the results
        assert_eq!(descriptor.keys().names(), &["species_center", "species_neighbor"]);
        assert_eq!(descriptor.keys().iter().collect::<Vec<_>>(), [
            &[-42, 1],
            &[1, -42],
            &[1, 1],
            &[1, 6],
            &[6, 1]
        ]);

        let mut expected_properties = LabelsBuilder::new(vec!["k"]);
        expected_properties.add(&[0]);
        let expected_properties = Arc::new(expected_properties.finish());

        /**********************************************************************/
        // O center, H neighbor
        let block = &descriptor.blocks()[0];
        let samples = &block.values().samples;
        assert_eq!(samples.names(), ["structure", "center"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [
            &[0, 0],
        ]);

        assert_eq!(block.values().properties, expected_properties);

        assert_eq!(block.values().data.as_array(), array![[2.0 / 2.0]].into_dyn());

        /**********************************************************************/
        // H center, O neighbor
        let block = &descriptor.blocks()[1];
        let samples = &block.values().samples;
        assert_eq!(samples.names(), ["structure", "center"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [
            &[0, 1], &[0, 2],
        ]);

        assert_eq!(block.values().properties, expected_properties);

        assert_eq!(block.values().data.as_array(), array![[1.0 / 2.0], [1.0 / 2.0]].into_dyn());

        /**********************************************************************/
        // H center, H neighbor
        let block = &descriptor.blocks()[2];
        let samples = &block.values().samples;
        assert_eq!(samples.names(), ["structure", "center"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [
            &[0, 1], &[0, 2],
        ]);

        assert_eq!(block.values().properties, expected_properties);

        assert_eq!(block.values().data.as_array(), array![[1.0 / 2.0], [1.0 / 2.0]].into_dyn());

        /**********************************************************************/
        // H center, C neighbor
        let block = &descriptor.blocks()[3];
        let samples = &block.values().samples;
        assert_eq!(samples.names(), ["structure", "center"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [
            &[1, 1],
        ]);

        assert_eq!(block.values().properties, expected_properties);

        assert_eq!(block.values().data.as_array(), array![[1.0 / 1.0]].into_dyn());

        /**********************************************************************/
        // C center, H neighbor
        let block = &descriptor.blocks()[4];
        let samples = &block.values().samples;
        assert_eq!(samples.names(), ["structure", "center"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), [
            &[1, 0],
        ]);

        assert_eq!(block.values().properties, expected_properties);

        assert_eq!(block.values().data.as_array(), array![[1.0 / 1.0]].into_dyn());
    }
}
// [property-test]

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::Calculator;
    use crate::systems::test_utils::{test_systems, test_system};

    // [partial-test]
    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 6,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        // build a list of samples to compute
        let mut samples = LabelsBuilder::new(vec![
            "structure", "center"
        ]);
        samples.add(&[0, 1]);
        samples.add(&[0, 2]);
        samples.add(&[1, 0]);
        samples.add(&[1, 2]);
        let samples = samples.finish();

        // create some properties. There is no need to order them in the same way
        // as the default calculator
        let mut properties = LabelsBuilder::new(vec!["k"]);
        properties.add(&[2]);
        properties.add(&[1]);
        properties.add(&[5]);
        let properties = properties.finish();

        // this function will check that selecting samples/properties or both will
        // not change the result of the calculation
        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &samples, &properties
        );
    }
    // [partial-test]

    // [finite-differences-test]
    #[test]
    fn finite_differences() {
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 7,
        }) as Box<dyn CalculatorBase>);

        let system = test_system("water");

        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-6,
            epsilon: 1e-20,
        };

        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }
    // [finite-differences-test]
}
