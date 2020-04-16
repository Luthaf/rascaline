use std::collections::BTreeSet;

use crate::system::System;
use super::{Indexes, IndexesBuilder, EnvironmentIndexes};


pub struct StructureSpeciesIdx;

impl EnvironmentIndexes for StructureSpeciesIdx {
    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        let mut indexes = IndexesBuilder::new(vec!["structure", "alpha"]);
        for (i_system, system) in systems.iter().enumerate() {
            for &species in system.species().iter().collect::<BTreeSet<_>>() {
                indexes.add(&[i_system, species]);
            }
        }
        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [&mut dyn System]) -> (Indexes, Option<Indexes>) {
        let mut gradients = IndexesBuilder::new(vec!["structure", "alpha", "atom", "spatial"]);
        for (i_system, system) in systems.iter().enumerate() {
            let species = system.species();
            for &alpha in species.iter().collect::<BTreeSet<_>>()  {
                for atom in 0..system.size() {
                    // only atoms with the same species participate to the gradient
                    if species[atom] == alpha {
                        gradients.add(&[i_system, alpha, atom, 0]);
                        gradients.add(&[i_system, alpha, atom, 1]);
                        gradients.add(&[i_system, alpha, atom, 2]);
                    }
                }
            }
        }
        return (self.indexes(systems), Some(gradients.finish()));
    }
}

pub struct PairSpeciesIdx {
    cutoff: f64,
}

impl PairSpeciesIdx {
    pub fn new(cutoff: f64) -> PairSpeciesIdx {
        assert!(cutoff > 0.0, "cutoff must be positive for PairSpeciesIdx");
        PairSpeciesIdx {
            cutoff: cutoff
        }
    }
}

impl EnvironmentIndexes for PairSpeciesIdx {
    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        // Accumulate indexes in a set first to ensure unicity of the indexes.
        // Else each neighbors of the same type for a given center would add a
        // new index for this center
        let mut set = BTreeSet::new();
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            let nl = system.neighbors();
            let species = system.species();
            nl.foreach_pair(&mut |i, j, _| {
                let species_i = species[i];
                let species_j = species[j];

                set.insert([i_system, i, species_i, species_j]);
                set.insert([i_system, j, species_j, species_i]);
            });
        }

        let mut indexes = IndexesBuilder::new(vec!["structure", "atom", "alpha", "beta"]);
        for idx in set {
            indexes.add(&idx);
        }
        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [&mut dyn System]) -> (Indexes, Option<Indexes>) {
        // this needs to deal with cutoff to only include atoms inside the
        // cutoff sphere
        unimplemented!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_systems;

    #[test]
    fn structure() {
        let mut systems = test_systems(vec!["methane", "methane", "water"]);
        let indexes = StructureSpeciesIdx.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 6);
        assert_eq!(indexes.names(), &["structure", "alpha"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[0, 1], &[0, 6],
            &[1, 1], &[1, 6],
            &[2, 1], &[2, 123456],
        ]);
    }

    #[test]
    fn structure_gradient() {
        let mut systems = test_systems(vec!["ch", "water"]);
        let (_, gradients) = StructureSpeciesIdx.with_gradients(&mut systems.get());
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 15);
        assert_eq!(gradients.names(), &["structure", "alpha", "atom", "spatial"]);

        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel in CH
            &[0, 1, 0, 0], &[0, 1, 0, 1], &[0, 1, 0, 2],
            // C channel in CH
            &[0, 6, 1, 0], &[0, 6, 1, 1], &[0, 6, 1, 2],
            // H channel in water
            &[1, 1, 1, 0], &[1, 1, 1, 1], &[1, 1, 1, 2],
            &[1, 1, 2, 0], &[1, 1, 2, 1], &[1, 1, 2, 2],
            // O channel in water
            &[1, 123456, 0, 0], &[1, 123456, 0, 1], &[1, 123456, 0, 2],
        ]);
    }

    #[test]
    fn pairs() {
        let mut systems = test_systems(vec!["ch", "water"]);
        let strategy = PairSpeciesIdx::new(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 7);
        assert_eq!(indexes.names(), &["structure", "atom", "alpha", "beta"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H in CH
            &[0, 0, 1, 6],
            // C in CH
            &[0, 1, 6, 1],
            // O in water
            &[1, 0, 123456, 1],
            // first H in water
            &[1, 1, 1, 1],
            &[1, 1, 1, 123456],
            // second H in water
            &[1, 2, 1, 1],
            &[1, 2, 1, 123456],
        ]);
    }

    #[test]
    #[ignore]
    fn pairs_gradient() {
        todo!()
    }
}
