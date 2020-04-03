use std::collections::{BTreeMap, BTreeSet};

use crate::system::System;
use super::{Indexes, IndexesBuilder, EnvironmentIndexes};


pub struct StructureSpeciesIdx;

impl EnvironmentIndexes for StructureSpeciesIdx {
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut indexes = IndexesBuilder::new(vec!["alpha", "structure"]);

        // List the systems containing a given species. The keys of the map are
        // species, and the values list of system id containing this species
        let mut systems_by_species = BTreeMap::new();
        for (i_system, system) in systems.iter().enumerate() {
            let all_species = system.species().iter().collect::<BTreeSet<_>>();
            for s in all_species {
                systems_by_species.entry(s).or_insert(vec![]).push(i_system)
            }
        }

        for (&species, systems) in systems_by_species {
            for i_system in systems {
                indexes.add(&[species, i_system]);
            }
        }

        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
        let indexes = self.indexes(systems);

        let mut values = Vec::new();
        for linear in 0..indexes.count() {
            let id = indexes.value(linear);
            let alpha = id[0];
            let structure = id[1];

            let system = &systems[structure];
            // only atoms with the same species participate in the gradient
            for (atom, &atom_species) in system.species().iter().enumerate() {
                if alpha == atom_species {
                    values.push((alpha, structure, atom));
                }
            }
        }

        let mut gradients = IndexesBuilder::new(vec!["spatial", "alpha", "structure", "atom"]);
        for spatial in 0..3 {
            for &(alpha, structure, atom) in values.iter() {
                gradients.add(&[spatial, alpha, structure, atom])
            }
        }

        return (indexes, Some(gradients.finish()));
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
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut set = BTreeSet::new();

        let all_species = systems.iter().flat_map(|s| s.species().iter().cloned()).collect::<BTreeSet<_>>();
        for &alpha in all_species.iter() {
            for &beta in all_species.iter() {
                for (i_system, system) in systems.iter_mut().enumerate() {
                    system.compute_neighbors(self.cutoff);
                    let nl = system.neighbors();
                    let species = system.species();
                    nl.foreach_pair(&mut |i, j, _| {
                        if species[i] == alpha && species[j] == beta {
                            set.insert([alpha, beta, i_system, i]);
                            if alpha == beta {
                                set.insert([alpha, beta, i_system, j]);
                            }
                        } else if species[j] == alpha && species[i] == beta {
                            set.insert([alpha, beta, i_system, j]);
                            if alpha == beta {
                                set.insert([alpha, beta, i_system, i]);
                            }
                        }
                    });
                }
            }
        }

        let mut indexes = IndexesBuilder::new(vec!["alpha", "beta", "structure", "center"]);
        for idx in set {
            indexes.add(&idx);
        }
        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
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
        let systems = &mut test_systems(vec!["methane", "methane", "water"]);
        let indexes = StructureSpeciesIdx.indexes(systems);
        assert_eq!(indexes.count(), 6);
        assert_eq!(indexes.names(), &["alpha", "structure"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[1, 0], &[1, 1], &[1, 2],
            &[6, 0], &[6, 1],
            &[123456, 2],
        ]);
    }

    #[test]
    fn structure_gradient() {
        let systems = &mut test_systems(vec!["ch", "water"]);
        let (_, gradients) = StructureSpeciesIdx.with_gradients(systems);
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 15);
        assert_eq!(gradients.names(), &["spatial", "alpha", "structure", "atom"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            &[0, 1, 0, 0], &[0, 1, 1, 1], &[0, 1, 1, 2], &[0, 6, 0, 1], &[0, 123456, 1, 0],
            &[1, 1, 0, 0], &[1, 1, 1, 1], &[1, 1, 1, 2], &[1, 6, 0, 1], &[1, 123456, 1, 0],
            &[2, 1, 0, 0], &[2, 1, 1, 1], &[2, 1, 1, 2], &[2, 6, 0, 1], &[2, 123456, 1, 0],
        ]);
    }

    #[test]
    fn pairs() {
        let systems = &mut test_systems(vec!["ch", "water"]);
        let strategy = PairSpeciesIdx::new(2.0);
        let indexes = strategy.indexes(systems);
        assert_eq!(indexes.count(), 7);
        assert_eq!(indexes.names(), &["alpha", "beta", "structure", "center"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H-H in water
            &[1, 1, 1, 1], &[1, 1, 1, 2],
            // H-C in CH
            &[1, 6, 0, 0],
            // H-O in water
            &[1, 123456, 1, 1], &[1, 123456, 1, 2],
            // C-H in CH
            &[6, 1, 0, 1],
            // O-H in water
            &[123456, 1, 1, 0],

        ]);
    }

    #[test]
    #[ignore]
    fn pairs_gradient() {
        todo!()
    }
}
