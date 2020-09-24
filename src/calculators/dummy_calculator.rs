use super::CalculatorBase;

use crate::descriptor::Descriptor;
use crate::descriptor::{IndexesBuilder, Indexes, EnvironmentIndexes, AtomEnvironment};
use crate::system::System;

/// A stupid calculator implementation used to test the API, and API binding to
/// C/Python/etc.
///
/// The calculator has two features: one containing the atom index +
/// `self.delta`, and the other one containg `x + y + z`.
#[doc(hidden)]
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct DummyCalculator {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Delta added to the atom index in the first feature
    pub delta: isize,
    /// Unused name parameter, to test passing string values
    pub name: String,
    /// Should we also compute gradients of the feature?
    pub gradients: bool,
}

impl CalculatorBase for DummyCalculator {
    fn name(&self) -> String {
        // abusing the name as description
        format!("dummy test calculator with cutoff: {} - delta: {} - name: {} - gradients: {}",
            self.cutoff, self.delta, self.name, self.gradients
        )
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(vec!["index_delta", "x_y_z"]);
        features.add(&[1, 0]);
        features.add(&[0, 1]);
        features.finish()
    }

    fn environments(&self) -> Box<dyn EnvironmentIndexes> {
        Box::new(AtomEnvironment::new(self.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
        assert_eq!(indexes.names(), &["index_delta", "x_y_z"]);
        for value in indexes.iter() {
            assert!(value == [0, 1] || value == [1, 0]);
        }
    }

    fn check_environments(&self, indexes: &Indexes, systems: &mut [&mut dyn System]) {
        assert_eq!(indexes.names(), ["structure", "center"]);
        // This could be made much faster by not recomputing the full list of
        // potential environments
        let allowed = self.environments().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid environment", value);
        }
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        for (i_sample, indexes) in descriptor.environments.iter().enumerate() {
            let mut current_structure = 0;
            let mut positions = systems[current_structure].positions();
            if let &[structure, atom] = indexes {
                if structure != current_structure {
                    current_structure = structure;
                    positions = systems[current_structure].positions();
                }

                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    if feature[0] == 1 {
                        descriptor.values[[i_sample, i_feature]] = atom as f64 + self.delta as f64;
                    } else if feature[1] == 1 {
                        let position = positions[atom];
                        descriptor.values[[i_sample, i_feature]] = position[0] + position[1] + position[2];
                    }
                }
            } else {
                panic!("got a bad environment indexes");
            }
        }

        if self.gradients {
            let gradients = descriptor.gradients.as_mut().expect("missing gradient values");
            let gradients_indexes = descriptor.gradients_indexes.as_ref().expect("missing gradient index");

            assert_eq!(gradients_indexes.names(), ["structure", "center", "neighbor", "spatial"]);

            for i_grad in 0..gradients_indexes.count() {
                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    if feature[0] == 1 {
                        gradients[[i_grad, i_feature]] = 0.0;
                    } else if feature[1] == 1 {
                        gradients[[i_grad, i_feature]] = 1.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::system::test_systems;
    use crate::{Descriptor, Calculator};
    use crate::descriptor::IndexesBuilder;

    use ndarray::{s, aview1};

    #[test]
    fn name() {
        let calculator = Calculator::new("dummy_calculator", "{
            \"cutoff\": 1.4,
            \"delta\": 9,
            \"name\": \"a long name\",
            \"gradients\": false
        }".to_owned()).unwrap();

        assert_eq!(
            calculator.name(),
            "dummy test calculator with cutoff: 1.4 - delta: 9 - name: a long name - gradients: false"
        );
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::new("dummy_calculator", "{
            \"cutoff\": 3.5,
            \"delta\": 9,
            \"name\": \"\",
            \"gradients\": false
        }".to_owned()).unwrap();

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.values.shape(), [3, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[9.0, 0.0]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[10.0, 0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[11.0, -1.3443999999999998]));
    }

    #[test]
    fn gradients() {
        let mut calculator = Calculator::new("dummy_calculator", "{
            \"cutoff\": 3.5,
            \"delta\": 0,
            \"name\": \"\",
            \"gradients\": true
        }".to_owned()).unwrap();

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [18, 2]);
        // 1 (structure) * 3 (centers) * 2 (neighbor per center) * 3 (spatial)
        assert_eq!(descriptor.gradients_indexes.unwrap().count(), 1 * 3 * 2 * 3);
        for i in 0..gradients.shape()[0] {
            assert_eq!(gradients.slice(s![i, ..]), aview1(&[0.0, 1.0]));
        }
    }

    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::new("dummy_calculator", "{
            \"cutoff\": 3.5,
            \"delta\": 9,
            \"name\": \"\",
            \"gradients\": true
        }".to_owned()).unwrap();

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();

        let mut samples = IndexesBuilder::new(vec!["structure", "center"]);
        samples.add(&[0, 1]);
        calculator.compute_partial(&mut systems.get(), &mut descriptor, Some(samples.finish()), None);

        assert_eq!(descriptor.values.shape(), [1, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[10.0, 0.16649999999999998]));


        let mut features = IndexesBuilder::new(vec!["index_delta", "x_y_z"]);
        features.add(&[0, 1]);
        calculator.compute_partial(&mut systems.get(), &mut descriptor, None, Some(features.finish()));

        assert_eq!(descriptor.values.shape(), [3, 1]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.0]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[-1.3443999999999998]));

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [18, 1]);
        for i in 0..gradients.shape()[0] {
            assert_eq!(gradients.slice(s![i, ..]), aview1(&[1.0]));
        }
    }
}
