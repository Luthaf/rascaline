use std::collections::HashMap;

use ndarray::{aview1, s};

use super::Calculator;

use crate::descriptor::{Descriptor, Indexes, IndexesBuilder, AtomSpeciesEnvironment};
use crate::system::System;
use crate::Error;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub struct SortedDistances {
    cutoff: f64,
    max_neighbors: usize,
}

impl SortedDistances {
    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(vec!["neighbor"]);
        for i in 0..self.max_neighbors {
            features.add(&[i]);
        }
        return features.finish();
    }
}

impl Calculator for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn parameters(&self) -> Result<String, Error> {
        Ok(serde_json::to_string(self)?)
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        if self.cutoff <= 0.0 || !self.cutoff.is_finite() {
            panic!("invalid cutoff value ({}) for {}", self.cutoff, self.name())
        }

        // setup the descriptor array
        let environments = AtomSpeciesEnvironment::new(self.cutoff);
        let features = self.features();
        descriptor.prepare(environments, features, systems, self.cutoff);
        assert_eq!(descriptor.environments.names(), &["structure", "center", "alpha", "beta"]);

        // index of the first entry of descriptor.values corresponding to
        // the current system
        let mut current = 0;
        for (i_system, system) in systems.iter_mut().enumerate() {
            // distance contains a vector of distances vector (one distance
            // vector for each center) for each pair of species in the system
            let mut distances = HashMap::new();
            for idx in &descriptor.environments {
                let alpha = idx[2];
                let beta = idx[3];
                distances.entry((alpha, beta)).or_insert(
                    vec![Vec::with_capacity(self.max_neighbors); system.size()]
                );
            }

            // Collect all distances around each center in `distances`
            system.compute_neighbors(self.cutoff);
            let species = system.species();
            for pair in system.pairs() {
                let i = pair.first;
                let j = pair.second;
                let d = pair.distance;

                let distances_vectors = distances.get_mut(&(species[i], species[j])).unwrap();
                distances_vectors[i].push(d);

                let distances_vectors = distances.get_mut(&(species[j], species[i])).unwrap();
                distances_vectors[j].push(d);
            }

            // Sort, resize to limit to at most `self.max_neighbors` values
            // and pad the distance vectors as needed
            for (_, vectors) in &mut distances {
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
                    if structure != i_system {
                        break;
                    }

                    let distance_vector = &distances.get(&(alpha, beta)).unwrap()[center];
                    descriptor.values.slice_mut(s![current, ..]).assign(&aview1(distance_vector));
                } else {
                    unreachable!();
                }
                current += 1;
            }
        }

        // did we get everything?
        assert_eq!(current, descriptor.environments.count());
    }
}

#[cfg(test)]
mod tests {
    use crate::system::test_systems;
    use crate::Descriptor;

    use ndarray::{s, aview1};

    use super::*;

    #[test]
    #[should_panic = "invalid cutoff value (-1.5) for sorted distances vector"]
    fn bad_cutoff_1() {
        let mut calculator = SortedDistances {
            cutoff: -1.5,
            max_neighbors: 3,
        };
        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);
    }

    #[test]
    #[should_panic = "invalid cutoff value (0) for sorted distances vector"]
    fn bad_cutoff_2() {
        let mut calculator = SortedDistances {
            cutoff: 0.0,
            max_neighbors: 3,
        };
        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);
    }

    #[test]
    fn name() {
        let calculator = SortedDistances {
            cutoff: 1.5,
            max_neighbors: 3,
        };

        assert_eq!(calculator.name(), "sorted distances vector");
    }

    #[test]
    fn values() {
        let mut calculator = SortedDistances {
            cutoff: 1.5,
            max_neighbors: 3,
        };

        let mut systems = test_systems(vec!["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
    }

    #[test]
    #[ignore]
    fn gradients() {
        unimplemented!()
    }
}
