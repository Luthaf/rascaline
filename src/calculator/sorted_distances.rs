use std::collections::HashMap;

use ndarray::{aview0, aview1, s};

use super::Calculator;

use crate::descriptor::{Descriptor, Indexes, IndexesBuilder, PairSpeciesIdx};
use crate::system::System;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize)]
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

    fn compute(&mut self, systems: &mut [Box<dyn System>]) -> Descriptor {
        // create the Descriptor array
        let environments = PairSpeciesIdx::new(self.cutoff);
        let features = self.features();
        let mut descriptor = Descriptor::new(environments, features, systems);

        descriptor.values.assign(&aview0(&self.cutoff));

        assert_eq!(descriptor.environments.names(), &["alpha", "beta", "structure", "center"]);

        // distance contains a vector of distances vector (one distance vector
        // for each center) for each pair of species in the systems
        let mut distances = HashMap::new();
        let centers = systems.iter().map(|s| s.size()).sum();
        for idx in &descriptor.environments {
            let alpha = idx[0];
            let beta = idx[1];
            distances.entry((alpha, beta)).or_insert(
                vec![Vec::with_capacity(self.max_neighbors); centers]
            );
        }

        // Get the first index of the first atom of each system
        let first_indexes = systems.iter()
            .scan(0, |acc, system| {
                *acc = *acc + system.size();
                Some(*acc)
            }).collect::<Vec<_>>();

        // Collect all distances around each center in `distances`
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            let nl = system.neighbors();
            let species = system.species();
            let start = first_indexes[i_system];
            nl.foreach_pair(&mut |i, j, d| {
                let distances_vectors = distances.get_mut(&(species[i], species[j])).unwrap();
                distances_vectors[start + i].push(d);

                let distances_vectors = distances.get_mut(&(species[j], species[i])).unwrap();
                distances_vectors[start + j].push(d);
            });
        }

        // Sort, resize to limit to at most `self.max_neighbors` values
        // and pad the distance vectors as needed
        for (_, vectors) in &mut distances {
            for vec in vectors {
                vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                vec.resize(self.max_neighbors, self.cutoff);
            }
        }

        // Copy the data in the descriptor array
        for (i, (alpha, beta, structure, center)) in descriptor.environments.iter().map(|idx| (idx[0], idx[1], idx[2], idx[3])).enumerate() {
            let env = first_indexes[structure] + center;
            let distance_vector = &distances.get(&(alpha, beta)).unwrap()[env];

            descriptor.values.slice_mut(s![i, ..]).assign(&aview1(distance_vector))
        }

        return descriptor;
    }
}
