use super::Calculator;

use crate::descriptor::{Descriptor, IndexesBuilder, AtomIdx};
use crate::system::System;

/// A stupid calculator implementation used to test the API, and API binding to
/// C/Python/etc. 
///
/// The calculator has two features: one containing the atom index +
/// `self.delta`, and the other one containg `x + y + z`.
#[doc(hidden)]
#[derive(Debug, Clone)]
#[derive(serde::Deserialize)]
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

impl Calculator for DummyCalculator {
    fn name(&self) -> String {
        // abusing the name as description
        format!("dummy test calculator with cutoff: {} - delta: {} - name: {} - gradients: {}",
            self.cutoff, self.delta, self.name, self.gradients
        )
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        let mut features = IndexesBuilder::new(vec!["index_delta", "x_y_z"]);
        features.add(&[1, 0]);
        features.add(&[0, 1]);
        let features = features.finish();

        let environments = AtomIdx::new(3.0);
        if self.gradients {
            descriptor.prepare_gradients(environments, features, systems, 0.0);
        } else {
            descriptor.prepare(environments, features, systems, 0.0);
        }

        assert_eq!(descriptor.environments.names(), ["structure", "atom"]);
        for (i, indexes) in descriptor.environments.iter().enumerate() {
            let mut current_structure = 0;
            let mut positions = systems[current_structure].positions();
            if let &[structure, atom] = indexes {
                if structure != current_structure {
                    current_structure = structure;
                    positions = systems[current_structure].positions();
                }

                descriptor.values[[i, 0]] = atom as f64 + self.delta as f64;
                let position = positions[atom];
                descriptor.values[[i, 1]] = position[0] + position[1] + position[2];
            } else {
                unreachable!();
            }
        }

        if self.gradients {
            let gradients = descriptor.gradients.as_mut().expect("missing gradient values");
            let gradients_indexes = descriptor.gradients_indexes.as_ref().expect("missing gradient index");

            assert_eq!(gradients_indexes.names(), ["structure", "atom", "neighbor", "spatial"]);

            for i in 0..gradients_indexes.count() {
                gradients[[i, 0]] = 0.0;
                gradients[[i, 1]] = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::system::test_systems;
    use crate::Descriptor;

    use ndarray::{s, aview1};

    use super::*;

    #[test]
    fn name() {
        let calculator = DummyCalculator {
            cutoff: 1.4,
            delta: 9,
            name: "a long name".into(),
            gradients: false,
        };

        assert_eq!(
            calculator.name(), 
            "dummy test calculator with cutoff: 1.4 - delta: 9 - name: a long name - gradients: false"
        );
    }

    #[test]
    fn values() {
        let mut calculator = DummyCalculator {
            cutoff: 3.5,
            delta: 9,
            name: "".into(),
            gradients: false,
        };

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[9.0, 0.0]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[10.0, 0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[11.0, -1.3443999999999998]));
    }

    #[test]
    fn gradients() {
        let mut calculator = DummyCalculator {
            cutoff: 3.5,
            delta: 0,
            name: "".into(),
            gradients: true,
        };

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.0, 0.0]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[1.0, 0.16649999999999998]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[2.0, -1.3443999999999998]));

        let gradients = descriptor.gradients.unwrap();
        // 1 (structure) * 3 (centers) * 2 (neighbor per center) * 3 (spatial)
        assert_eq!(descriptor.gradients_indexes.unwrap().count(), 1 * 3 * 2 * 3);
        for i in 0..gradients.shape()[0] {
            assert_eq!(gradients.slice(s![i, ..]), aview1(&[0.0, 1.0]));
        }
    }
}