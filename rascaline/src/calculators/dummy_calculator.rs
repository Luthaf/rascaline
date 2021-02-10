use super::CalculatorBase;

use crate::descriptor::Descriptor;
use crate::descriptor::{IndexesBuilder, Indexes, IndexValue, EnvironmentIndexes, AtomEnvironment};
use crate::system::System;

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

    fn get_parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn features_names(&self) -> Vec<&str> {
        vec!["index_delta", "x_y_z", "float"]
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(self.features_names());
        features.add(&[IndexValue::from(1_usize), IndexValue::from(0_isize), IndexValue::from(1.2)]);
        features.add(&[IndexValue::from(0_usize), IndexValue::from(1_isize), IndexValue::from(3.2)]);
        features.finish()
    }

    fn environments(&self) -> Box<dyn EnvironmentIndexes> {
        Box::new(AtomEnvironment::new(self.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
        assert_eq!(indexes.names(), self.features_names());
        let first = [IndexValue::from(1_usize), IndexValue::from(0_isize), IndexValue::from(1.2)];
        let second = [IndexValue::from(0_usize), IndexValue::from(1_isize), IndexValue::from(3.2)];
        for value in indexes.iter() {
            assert!(value == first || value == second);
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

    #[allow(clippy::clippy::cast_precision_loss)]
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        for (i_sample, indexes) in descriptor.environments.iter().enumerate() {
            let i_system = indexes[0].usize();
            let center = indexes[1].usize();

            for (i_feature, feature) in descriptor.features.iter().enumerate() {
                if feature[0].isize() == 1 {
                    descriptor.values[[i_sample, i_feature]] = center as f64 + self.delta as f64;
                } else if feature[1].isize() == 1 {
                    let system = &mut *systems[i_system];
                    system.compute_neighbors(self.cutoff);

                    let positions = system.positions();
                    let mut sum = positions[center][0] + positions[center][1] + positions[center][2];
                    for pair in system.pairs() {
                        if pair.first == center {
                            sum += positions[pair.second][0] + positions[pair.second][1] + positions[pair.second][2];
                        }

                        if pair.second == center {
                            sum += positions[pair.first][0] + positions[pair.first][1] + positions[pair.first][2];
                        }
                    }

                    descriptor.values[[i_sample, i_feature]] = sum;
                }
            }
        }

        if self.gradients {
            let gradients = descriptor.gradients.as_mut().expect("missing gradient values");
            let gradients_indexes = descriptor.gradients_indexes.as_ref().expect("missing gradient index");

            assert_eq!(gradients_indexes.names(), ["structure", "center", "neighbor", "spatial"]);

            for i_grad in 0..gradients_indexes.count() {
                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    if feature[0].isize() == 1 {
                        gradients[[i_grad, i_feature]] = 0.0;
                    } else if feature[1].isize() == 1 {
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
    use crate::{CalculationOptions, SelectedIndexes};
    use crate::descriptor::{IndexesBuilder, IndexValue};

    use ndarray::{s, aview1};

    use super::DummyCalculator;
    use super::super::CalculatorBase;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.4,
            delta: 9,
            name: "a long name".into(),
            gradients: false,
        }) as Box<dyn CalculatorBase>);

        assert_eq!(
            calculator.name(),
            "dummy test calculator with cutoff: 1.4 - delta: 9 - name: a long name - gradients: false"
        );

        assert_eq!(
            calculator.parameters(),
            "{\"cutoff\":1.4,\"delta\":9,\"name\":\"a long name\",\"gradients\":false}"
        );
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.0,
            delta: 9,
            name: "".into(),
            gradients: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor, Default::default()).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[9.0, -1.1778999999999997]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[10.0, 0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[11.0, -1.3443999999999998]));
    }

    #[test]
    fn gradients() {
        let mut calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.0,
            delta: 9,
            name: "".into(),
            gradients: true,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor, Default::default()).unwrap();

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [12, 2]);
        for i in 0..gradients.shape()[0] {
            assert_eq!(gradients.slice(s![i, ..]), aview1(&[0.0, 1.0]));
        }
    }

    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::from(Box::new(DummyCalculator{
            cutoff: 1.0,
            delta: 9,
            name: "".into(),
            gradients: true,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();

        let mut samples = IndexesBuilder::new(vec!["structure", "center"]);
        samples.add(&[IndexValue::from(0_usize), IndexValue::from(1_usize)]);

        let options = CalculationOptions {
            selected_samples: SelectedIndexes::Some(samples.finish()),
            selected_features: SelectedIndexes::All,
            ..Default::default()
        };
        calculator.compute(&mut systems.get(), &mut descriptor, options).unwrap();

        assert_eq!(descriptor.values.shape(), [1, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[10.0, 0.16649999999999998]));

        let mut features = IndexesBuilder::new(vec!["index_delta", "x_y_z", "float"]);
        features.add(&[IndexValue::from(0_usize), IndexValue::from(1_isize), IndexValue::from(3.2)]);
        let options = CalculationOptions {
            selected_samples: SelectedIndexes::All,
            selected_features: SelectedIndexes::Some(features.finish()),
            ..Default::default()
        };
        calculator.compute(&mut systems.get(), &mut descriptor, options).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 1]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[-1.1778999999999997]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[-1.3443999999999998]));

        let gradients = descriptor.gradients.unwrap();
        assert_eq!(gradients.shape(), [12, 1]);
        for i in 0..gradients.shape()[0] {
            assert_eq!(gradients.slice(s![i, ..]), aview1(&[1.0]));
        }
    }
}
