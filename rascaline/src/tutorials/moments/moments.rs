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
        "geometric moments".to_string()
    }

    fn get_parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn compute_gradients(&self) -> bool {
        self.gradients
    }

    fn features_names(&self) -> Vec<&str> {
        vec!["k"]
    }

    fn features(&self) -> Indexes {
        let mut builder = IndexesBuilder::new(self.features_names());
        for k in 0..=self.max_moment {
            builder.add(&[IndexValue::from(k)]);
        }

        return builder.finish();
    }

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

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        Box::new(TwoBodiesSpeciesSamples::new(self.cutoff))
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        assert_eq!(descriptor.samples.names(), self.samples_builder().names());
        assert_eq!(descriptor.features.names(), self.features_names());

        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            for pair in system.pairs()? {
                let first_sample = [
                    IndexValue::from(i_system),
                    IndexValue::from(pair.first),
                    IndexValue::from(species[pair.first]),
                    IndexValue::from(species[pair.second]),
                ];

                let second_sample = [
                    IndexValue::from(i_system),
                    IndexValue::from(pair.second),
                    IndexValue::from(species[pair.second]),
                    IndexValue::from(species[pair.first]),
                ];

                let first_sample_position = descriptor.samples.position(&first_sample);
                let second_sample_position = descriptor.samples.position(&second_sample);

                // skip calculation if neither of the samples is present
                if first_sample_position.is_none() && second_sample_position.is_none() {
                    continue;
                }

                let n_neighbors_first = system.pairs_containing(pair.first)?.len() as f64;
                let n_neighbors_second = system.pairs_containing(pair.second)?.len() as f64;

                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    let k = feature[0].usize();
                    let moment = f64::powi(pair.distance, k as i32);

                    if let Some(i_first_sample) = first_sample_position {
                        descriptor.values[[i_first_sample, i_feature]] += moment / n_neighbors_first;
                    }

                    if let Some(i_second_sample) = second_sample_position {
                        descriptor.values[[i_second_sample, i_feature]] += moment / n_neighbors_second;
                    }
                }

                if self.gradients {
                    let gradients_samples = descriptor.gradients_samples.as_ref().expect("missing gradient samples");
                    let gradients = descriptor.gradients.as_mut().expect("missing gradient storage");

                    let first_gradient = [
                        IndexValue::from(i_system),
                        IndexValue::from(pair.first),
                        IndexValue::from(species[pair.first]),
                        IndexValue::from(species[pair.second]),
                        IndexValue::from(pair.second),
                        IndexValue::from(0),
                    ];

                    let first_gradient_self = [
                        IndexValue::from(i_system),
                        IndexValue::from(pair.first),
                        IndexValue::from(species[pair.first]),
                        IndexValue::from(species[pair.second]),
                        IndexValue::from(pair.first),
                        IndexValue::from(0),
                    ];

                    let second_gradient = [
                        IndexValue::from(i_system),
                        IndexValue::from(pair.second),
                        IndexValue::from(species[pair.second]),
                        IndexValue::from(species[pair.first]),
                        IndexValue::from(pair.first),
                        IndexValue::from(0),
                    ];

                    let second_gradient_self = [
                        IndexValue::from(i_system),
                        IndexValue::from(pair.second),
                        IndexValue::from(species[pair.second]),
                        IndexValue::from(species[pair.first]),
                        IndexValue::from(pair.second),
                        IndexValue::from(0),
                    ];

                    let first_gradient_position = gradients_samples.position(&first_gradient);
                    let first_gradient_self_position = gradients_samples.position(&first_gradient_self);
                    let second_gradient_position = gradients_samples.position(&second_gradient);
                    let second_gradient_self_position = gradients_samples.position(&second_gradient_self);

                    for (i_feature, feature) in descriptor.features.iter().enumerate() {
                        let k = feature[0].usize();
                        let grad_factor = k as f64 * f64::powi(pair.distance, (k as i32) - 2);

                        let grad_x = pair.vector[0] * grad_factor;
                        let grad_y = pair.vector[1] * grad_factor;
                        let grad_z = pair.vector[2] * grad_factor;

                        if let Some(i_first) = first_gradient_position {
                            gradients[[i_first + 0, i_feature]] += grad_x / n_neighbors_first;
                            gradients[[i_first + 1, i_feature]] += grad_y / n_neighbors_first;
                            gradients[[i_first + 2, i_feature]] += grad_z / n_neighbors_first;
                        }

                        if let Some(i_first_self) = first_gradient_self_position {
                            gradients[[i_first_self + 0, i_feature]] += -grad_x / n_neighbors_first;
                            gradients[[i_first_self + 1, i_feature]] += -grad_y / n_neighbors_first;
                            gradients[[i_first_self + 2, i_feature]] += -grad_z / n_neighbors_first;
                        }

                        if let Some(i_second) = second_gradient_position {
                            gradients[[i_second + 0, i_feature]] += -grad_x / n_neighbors_second;
                            gradients[[i_second + 1, i_feature]] += -grad_y / n_neighbors_second;
                            gradients[[i_second + 2, i_feature]] += -grad_z / n_neighbors_second;
                        }

                        if let Some(i_second_self) = second_gradient_self_position {
                            gradients[[i_second_self + 0, i_feature]] += grad_x / n_neighbors_second;
                            gradients[[i_second_self + 1, i_feature]] += grad_y / n_neighbors_second;
                            gradients[[i_second_self + 2, i_feature]] += grad_z / n_neighbors_second;
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

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    #[test]
    fn zeroth_moment() {
        // Create a Calculator wrapping a GeometricMoments instance
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 2,
            gradients: false,
        }) as Box<dyn CalculatorBase>);

        // create a bunch of systems in a format compatible with `calculator.compute`.
        // Available systems include "water" and "methane" for the corresponding
        // molecules, and "CH" for a basic 2 atoms system.
        let mut systems = test_systems(&["water", "CH"]);

        // run the calculation using default parameters
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems, &mut descriptor, Default::default()).unwrap();

        // check the results
        assert_eq!(descriptor.features.names(), &["k"]);
        assert_eq!(descriptor.features.count(), 3);
        assert_eq!(&descriptor.features[0], &[v(0)]);
        assert_eq!(&descriptor.features[1], &[v(1)]);
        assert_eq!(&descriptor.features[2], &[v(2)]);

        assert_eq!(
            descriptor.samples.names(),
            ["structure", "center", "species_center", "species_neighbor"]
        );
        assert_eq!(descriptor.samples.count(), 7);

        // H neighbors around the oxygen atom in the water molecule
        assert_eq!(&descriptor.samples[0], &[
            v(0),        // structure 0
            v(0),        // center 0
            v(123456),   // species for oxygen in the test system
            v(1),        // species for hydrogen in the test system
        ]);
        assert_eq!(descriptor.values[[0, 0]], 2.0 / 2.0);

        // H neighbors around the first H atom in the water molecule
        assert_eq!(&descriptor.samples[1], &[v(0), v(1), v(1), v(1)]);
        assert_eq!(descriptor.values[[1, 0]], 1.0 / 2.0);

        // O neighbors around the first H atom in the water molecule
        assert_eq!(&descriptor.samples[2], &[v(0), v(1), v(1), v(123456)]);
        assert_eq!(descriptor.values[[2, 0]], 1.0 / 2.0);

        // H neighbors around the second H atom in the water molecule
        assert_eq!(&descriptor.samples[3], &[v(0), v(2), v(1), v(1)]);
        assert_eq!(descriptor.values[[3, 0]], 1.0 / 2.0);

        // O neighbors around the second H atom in the water molecule
        assert_eq!(&descriptor.samples[4], &[v(0), v(2), v(1), v(123456)]);
        assert_eq!(descriptor.values[[4, 0]], 1.0 / 2.0);

        // C neighbors around the H atom in the CH molecule
        assert_eq!(&descriptor.samples[5], &[v(1), v(0), v(1), v(6)]);
        assert_eq!(descriptor.values[[5, 0]], 1.0 / 1.0);

        // H neighbors around the C atom in the CH molecule
        assert_eq!(&descriptor.samples[6], &[v(1), v(1), v(6), v(1)]);
        assert_eq!(descriptor.values[[6, 0]], 1.0 / 1.0);
    }
}
// [property-test]

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::Calculator;
    use crate::systems::test_utils::{test_systems, test_system};

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    // [partial-test]
    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 6,
            gradients: true,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        // build a list of samples to compute
        let mut samples = IndexesBuilder::new(vec![
            "structure", "center", "species_center", "species_neighbor"
        ]);
        samples.add(&[v(0), v(1), v(1), v(1)]);
        samples.add(&[v(0), v(2), v(1), v(123456)]);
        samples.add(&[v(1), v(0), v(6), v(1)]);
        samples.add(&[v(1), v(2), v(1), v(1)]);
        let samples = samples.finish();

        // create some features. There is no need to order them in the same way
        // as the default calculator
        let mut features = IndexesBuilder::new(vec!["k"]);
        features.add(&[v(2)]);
        features.add(&[v(1)]);
        features.add(&[v(5)]);
        let features = features.finish();

        // this function will check that selecting samples/features or both will
        // not change the result of the calculation
        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, samples, features
        );
    }
    // [partial-test]

    // [finite-differences-test]
    #[test]
    fn finite_differences() {
        let mut calculator = Calculator::from(Box::new(GeometricMoments{
            cutoff: 3.4,
            max_moment: 7,
            gradients: true,
        }) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        crate::calculators::tests_utils::finite_difference(calculator, system);
    }
    // [finite-differences-test]
}
