use std::collections::HashMap;

use ndarray::{aview1, s};

use super::CalculatorBase;

use crate::descriptor::Descriptor;
use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
use crate::descriptor::{EnvironmentIndexes, AtomSpeciesEnvironment};
use crate::system::System;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct SortedDistances {
    cutoff: f64,
    max_neighbors: usize,
}

impl CalculatorBase for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(vec!["neighbor"]);
        for i in 0..self.max_neighbors {
            features.add(&[IndexValue::from(i)]);
        }
        return features.finish();
    }

    fn environments(&self) -> Box<dyn EnvironmentIndexes> {
        Box::new(AtomSpeciesEnvironment::new(self.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        false
    }

    fn check_features(&self, indexes: &Indexes) {
        assert_eq!(indexes.names(), &["neighbor"]);
        for value in indexes.iter() {
            assert!(value[0].usize() < self.max_neighbors);
        }
    }

    fn check_environments(&self, indexes: &Indexes, systems: &mut [&mut dyn System]) {
        assert_eq!(indexes.names(), &["structure", "center", "alpha", "beta"]);
        // This could be made much faster by not recomputing the full list of
        // potential environments
        let allowed = self.environments().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid environment", value);
        }
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        let all_features = descriptor.features.count() == self.max_neighbors;
        let mut requested_features = Vec::new();
        if !all_features {
            for feature in descriptor.features.iter() {
                let neighbor = feature[0];
                requested_features.push(neighbor);
            }
        }

        // index of the first entry of descriptor.values corresponding to
        // the current system
        let mut current = 0;
        for (i_system, system) in systems.iter_mut().enumerate() {
            // distance contains a vector of distances vector (one distance
            // vector for each center) for each pair of species in the system
            let mut distances = HashMap::new();
            for idx in &descriptor.environments {
                let alpha = idx[2].usize();
                let beta = idx[3].usize();
                distances.entry((alpha, beta)).or_insert_with(
                    || vec![Vec::with_capacity(self.max_neighbors); system.size()]
                );
            }

            // Collect all distances around each center in `distances`
            system.compute_neighbors(self.cutoff);
            let species = system.species();
            for pair in system.pairs() {
                let i = pair.first;
                let j = pair.second;
                let d = pair.distance;

                if let Some(distances) = distances.get_mut(&(species[i], species[j])) {
                    distances[i].push(d);
                }

                if let Some(distances) = distances.get_mut(&(species[j], species[i])) {
                    distances[j].push(d);
                }
            }

            // Sort, resize to limit to at most `self.max_neighbors` values
            // and pad the distance vectors as needed
            for vectors in distances.iter_mut().map(|(_, vectors)| vectors) {
                for vec in vectors {
                    vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    vec.resize(self.max_neighbors, self.cutoff);
                }
            }

            loop {
                if current == descriptor.environments.count() {
                    break;
                }

                // Copy the data in the descriptor array, until we find the
                // next system
                if let [structure, center, alpha, beta] = descriptor.environments[current] {
                    if structure.usize() != i_system {
                        break;
                    }

                    let distance_vector = &distances.get(&(alpha.usize(), beta.usize())).unwrap()[center.usize()];
                    if all_features {
                        descriptor.values.slice_mut(s![current, ..]).assign(&aview1(distance_vector));
                    } else {
                        // Only assign the requested values
                        for (i, &neighbor) in requested_features.iter().enumerate() {
                            descriptor.values[[current, i]] = distance_vector[neighbor.usize()];
                        }
                    }
                } else {
                    unreachable!();
                }
                current += 1;
            }
        }

        // sanity check: did we get all environment in the above loop?
        assert_eq!(current, descriptor.environments.count());
    }
}

#[cfg(test)]
mod tests {
    use crate::system::test_systems;
    use crate::{Descriptor, Calculator};
    use crate::descriptor::{IndexesBuilder, IndexValue};

    use ndarray::{s, aview1};

    #[test]
    fn name() {
        let calculator = Calculator::new("sorted_distances", "{
            \"cutoff\": 1.5,
            \"max_neighbors\": 3
        }".to_owned()).unwrap();

        assert_eq!(calculator.name(), "sorted distances vector");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::new("sorted_distances", "{
            \"cutoff\": 1.5,
            \"max_neighbors\": 3
        }".to_owned()).unwrap();

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.values.shape(), [3, 3]);

        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
    }

    #[test]
    #[ignore]
    fn gradients() {
        unimplemented!()
    }

    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::new("sorted_distances", "{
            \"cutoff\": 1.5,
            \"max_neighbors\": 3
        }".to_owned()).unwrap();

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();

        let mut samples = IndexesBuilder::new(vec!["structure", "center", "alpha", "beta"]);
        samples.add(&[
            IndexValue::from(0_usize), IndexValue::from(1_usize),
            IndexValue::from(1_usize), IndexValue::from(123456_usize)
        ]);
        calculator.compute_partial(&mut systems.get(), &mut descriptor, Some(samples.finish()), None);

        assert_eq!(descriptor.values.shape(), [1, 3]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));

        let mut features = IndexesBuilder::new(vec!["neighbor"]);
        features.add(&[IndexValue::from(0_usize)]);
        features.add(&[IndexValue::from(2_usize)]);
        calculator.compute_partial(&mut systems.get(), &mut descriptor, None, Some(features.finish()));

        assert_eq!(descriptor.values.shape(), [3, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5]));
    }
}
