use std::collections::BTreeSet;

use crate::system::System;
use super::{Indexes, IndexesBuilder, EnvironmentIndexes};

/// `StructureSpeciesEnvironment` is used to represents environments corresponding to
/// full structures, where each chemical species is represented separatedly.
///
/// The base set of indexes contains `structure` and `alpha` (i.e. chemical
/// species); the  gradient indexes also contains the `atom` inside the
/// structure with respect to which the gradient is taken and the `spatial`
/// (i.e. x/y/z) index.
pub struct StructureSpeciesEnvironment;

impl EnvironmentIndexes for StructureSpeciesEnvironment {
    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        let mut indexes = IndexesBuilder::new(vec!["structure", "alpha"]);
        for (i_system, system) in systems.iter().enumerate() {
            for &species in system.species().iter().collect::<BTreeSet<_>>() {
                indexes.add(&[i_system, species]);
            }
        }
        return indexes.finish();
    }

    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), ["structure", "alpha"]);

        let mut gradients = IndexesBuilder::new(vec!["structure", "alpha", "atom", "spatial"]);
        for value in samples.iter() {
            let i_system = value[0];
            let alpha = value[1];

            let system = &systems[i_system];
            let species = system.species();
            for atom in 0..system.size() {
                // only atoms with the same species participate to the gradient
                if species[atom] == alpha {
                    gradients.add(&[i_system, alpha, atom, 0]);
                    gradients.add(&[i_system, alpha, atom, 1]);
                    gradients.add(&[i_system, alpha, atom, 2]);
                }
            }
        }

        return Some(gradients.finish());
    }
}

/// `AtomSpeciesEnvironment` is used to represents atom-centered environments, where
/// each atom in a structure is described with a feature vector based on other
/// atoms inside a sphere centered on the central atom. These environments
/// include chemical species information.
///
/// The base set of indexes contains `structure`, `center` (i.e. central atom
/// index inside the structure), `alpha` (specie of the central atom) and `beta`
/// (species of the neighboring atom); the gradient indexes also contains the
/// `neighbor` inside the spherical cutoff with respect to which the gradient is
/// taken and the `spatial` (i.e x/y/z) index.
pub struct AtomSpeciesEnvironment {
    cutoff: f64,
}

impl AtomSpeciesEnvironment {
    /// Create a nex `AtomSpeciesEnvironment` with the goven `cutoff`
    pub fn new(cutoff: f64) -> AtomSpeciesEnvironment {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for AtomSpeciesEnvironment");
        AtomSpeciesEnvironment { cutoff }
    }
}

impl EnvironmentIndexes for AtomSpeciesEnvironment {
    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        // Accumulate indexes in a set first to ensure unicity of the indexes
        // even if their are multiple neighbors of the same specie around a
        // given center
        let mut set = BTreeSet::new();
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            let species = system.species();
            for pair in system.pairs() {
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                set.insert([i_system, pair.first, species_first, species_second]);
                set.insert([i_system, pair.second, species_second, species_first]);
            };
        }

        let mut indexes = IndexesBuilder::new(vec!["structure", "center", "alpha", "beta"]);
        for idx in set {
            indexes.add(&idx);
        }
        return indexes.finish();
    }

    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), ["structure", "center", "alpha", "beta"]);

        let requested_systems = samples.iter().map(|v| v[0]).collect::<BTreeSet<_>>();

        let mut set = BTreeSet::new();
        for i_system in requested_systems {
            let system = &mut systems[i_system];
            system.compute_neighbors(self.cutoff);
            let species = system.species();

            let requested_centers = samples.iter()
                .filter(|v| v[0] == i_system)
                .map(|v| v[1])
                .collect::<Vec<_>>();

            for pair in system.pairs() {
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                if requested_centers.contains(&pair.first) {
                    set.insert([i_system, pair.first, species_first, species_second, pair.second, 0]);
                    set.insert([i_system, pair.first, species_first, species_second, pair.second, 1]);
                    set.insert([i_system, pair.first, species_first, species_second, pair.second, 2]);
                }

                if requested_centers.contains(&pair.second) {
                    set.insert([i_system, pair.second, species_second, species_first, pair.first, 0]);
                    set.insert([i_system, pair.second, species_second, species_first, pair.first, 1]);
                    set.insert([i_system, pair.second, species_second, species_first, pair.first, 2]);
                }
            }
        }

        let mut gradients = IndexesBuilder::new(vec!["structure", "center", "alpha", "beta", "neighbor", "spatial"]);
        for idx in set {
            gradients.add(&idx);
        }

        return Some(gradients.finish());
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_systems;

    #[test]
    fn structure() {
        let mut systems = test_systems(vec!["methane", "methane", "water"]);
        let indexes = StructureSpeciesEnvironment.indexes(&mut systems.get());
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
        let (_, gradients) = StructureSpeciesEnvironment.with_gradients(&mut systems.get());
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
    fn atoms() {
        let mut systems = test_systems(vec!["ch", "water"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 7);
        assert_eq!(indexes.names(), &["structure", "center", "alpha", "beta"]);
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
    fn atoms_gradient() {
        let mut systems = test_systems(vec!["ch", "water"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let (_, gradients) = strategy.with_gradients(&mut systems.get());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "center", "alpha", "beta", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-C channel in CH
            &[0, 0, 1, 6, 1, 0], &[0, 0, 1, 6, 1, 1], &[0, 0, 1, 6, 1, 2],
            // C-H channel in CH
            &[0, 1, 6, 1, 0, 0], &[0, 1, 6, 1, 0, 1], &[0, 1, 6, 1, 0, 2],
            // O-H channel in water
            &[1, 0, 123456, 1, 1, 0], &[1, 0, 123456, 1, 1, 1], &[1, 0, 123456, 1, 1, 2],
            &[1, 0, 123456, 1, 2, 0], &[1, 0, 123456, 1, 2, 1], &[1, 0, 123456, 1, 2, 2],
            // H-H channel in water, 1st atom
            &[1, 1, 1, 1, 2, 0], &[1, 1, 1, 1, 2, 1], &[1, 1, 1, 1, 2, 2],
            // H-O channel in water, 1st atom
            &[1, 1, 1, 123456, 0, 0], &[1, 1, 1, 123456, 0, 1], &[1, 1, 1, 123456, 0, 2],
            // H-H channel in water, 2nd atom
            &[1, 2, 1, 1, 1, 0], &[1, 2, 1, 1, 1, 1], &[1, 2, 1, 1, 1, 2],
            // H-O channel in water, 2nd atom
            &[1, 2, 1, 123456, 0, 0], &[1, 2, 1, 123456, 0, 1], &[1, 2, 1, 123456, 0, 2],
        ]);
    }
}
