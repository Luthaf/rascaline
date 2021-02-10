use std::collections::BTreeSet;

use indexmap::IndexSet;
use itertools::Itertools;

use crate::system::System;
use super::{EnvironmentIndexes, Indexes, IndexesBuilder, IndexValue};

/// `StructureSpeciesEnvironment` is used to represents environments
/// corresponding to full structures, where each chemical species is represented
/// separately.
///
/// The base set of indexes contains `structure` and `species` the  gradient
/// indexes also contains the `atom` inside the structure with respect to which
/// the gradient is taken and the `spatial` (i.e. x/y/z) index.
pub struct StructureSpeciesEnvironment;

impl EnvironmentIndexes for StructureSpeciesEnvironment {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "species"]
    }

    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        let mut indexes = IndexesBuilder::new(self.names());
        for (i_system, system) in systems.iter().enumerate() {
            for &species in system.species().iter().collect::<BTreeSet<_>>() {
                indexes.add(&[
                    IndexValue::from(i_system), IndexValue::from(species)
                ]);
            }
        }
        return indexes.finish();
    }

    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), self.names());

        let mut gradients = IndexesBuilder::new(vec!["structure", "species", "atom", "spatial"]);
        for value in samples.iter() {
            let i_system = value[0];
            let alpha = value[1];

            let system = &systems[i_system.usize()];
            let species = system.species();
            for (i_atom, &species) in species.iter().enumerate() {
                // only atoms with the same species participate to the gradient
                if species == alpha.usize() {
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(0_usize)]);
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(1_usize)]);
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(2_usize)]);
                }
            }
        }

        return Some(gradients.finish());
    }
}

/// `AtomSpeciesEnvironment` is used to represents atom-centered environments,
/// where each atom in a structure is described with a feature vector based on
/// other atoms inside a sphere centered on the central atom. These environments
/// include chemical species information.
///
/// The base set of indexes contains `structure`, `center` (i.e. central atom
/// index inside the structure), `species_center` and `species_neighbor`; the
/// gradient indexes also contains the `neighbor` inside the spherical cutoff
/// with respect to which the gradient is taken and the `spatial` (i.e x/y/z)
/// index.
pub struct AtomSpeciesEnvironment {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
    /// Is the central atom considered to be its own neighbor?
    self_contribution: bool,
}

impl AtomSpeciesEnvironment {
    /// Create a new `AtomSpeciesEnvironment` with the given `cutoff`, excluding
    /// self contributions.
    pub fn new(cutoff: f64) -> AtomSpeciesEnvironment {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for AtomSpeciesEnvironment");
        AtomSpeciesEnvironment {
            cutoff: cutoff,
            self_contribution: false,
        }
    }

    /// Create a new `AtomSpeciesEnvironment` with the given `cutoff`, including
    /// self contributions.
    pub fn with_self_contribution(cutoff: f64) -> AtomSpeciesEnvironment {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for AtomSpeciesEnvironment");
        AtomSpeciesEnvironment {
            cutoff: cutoff,
            self_contribution: true,
        }
    }
}

impl EnvironmentIndexes for AtomSpeciesEnvironment {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center", "species_center", "species_neighbor"]
    }

    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        // Accumulate indexes in a set first to ensure uniqueness of the indexes
        // even if their are multiple neighbors of the same specie around a
        // given center
        let mut set = BTreeSet::new();
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            let species = system.species();
            for pair in system.pairs() {
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                set.insert((i_system, pair.first, species_first, species_second));
                set.insert((i_system, pair.second, species_second, species_first));
            };

            if self.self_contribution {
                for (center, &species) in species.iter().enumerate() {
                    set.insert((i_system, center, species, species));
                }
            }
        }

        let mut indexes = IndexesBuilder::new(self.names());
        for (s, c, a, b) in set {
            indexes.add(&[
                IndexValue::from(s), IndexValue::from(c), IndexValue::from(a), IndexValue::from(b)
            ]);
        }
        return indexes.finish();
    }

    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), self.names());

        // We need IndexSet to yield the indexes in the right order, i.e. the
        // order corresponding to whatever was passed in `samples`
        let mut indexes = IndexSet::new();
        for requested in samples {
            let i_system = requested[0];
            let center = requested[1].usize();
            let species_neighbor = requested[3].usize();

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff);

            let species = system.species();

            // FIXME: this will always be 0, but is required for Descriptor.densify
            if self.self_contribution && species[center] == species_neighbor {
                indexes.insert((i_system, center, species_neighbor, species_neighbor, center));
            }

            for pair in system.pairs_containing(center) {
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                if pair.first == center && species_second == species_neighbor {
                    indexes.insert((i_system, pair.first, species_first, species_second, pair.second));
                } else if pair.second == center && species_first == species_neighbor {
                    indexes.insert((i_system, pair.second, species_second, species_first, pair.first));
                }
            }
        }

        let mut gradients = IndexesBuilder::new(vec![
            "structure", "center", "species_center", "species_neighbor",
            "neighbor", "spatial"
        ]);
        for (system, c, a, b, n) in indexes {
            let center = IndexValue::from(c);
            let alpha = IndexValue::from(a);
            let beta = IndexValue::from(b);
            let neighbor = IndexValue::from(n);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(0_usize)]);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(1_usize)]);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(2_usize)]);
        }

        return Some(gradients.finish());
    }
}

/// `ThreeBodiesSpecies` is used to represents atom-centered environments
/// representing three body atomic density correlation; where the three bodies
/// include the central atom and two neighbors. These environments include
/// chemical species information.
///
/// The base set of indexes contains `structure`, `center` (i.e. central atom
/// index inside the structure), `species_center`, `species_neighbor_1` and
/// `species_neighbor2`; the gradient indexes also contains the `neighbor`
/// inside the spherical cutoff with respect to which the gradient is taken and
/// the `spatial` (i.e x/y/z) index.
pub struct ThreeBodiesSpeciesEnvironment {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
    /// Is the central atom considered to be its own neighbor?
    self_contribution: bool,
}

impl ThreeBodiesSpeciesEnvironment {
    /// Create a new `ThreeBodiesSpeciesEnvironment` with the given `cutoff`, excluding
    /// self contributions.
    pub fn new(cutoff: f64) -> ThreeBodiesSpeciesEnvironment {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for ThreeBodiesSpeciesEnvironment");
        ThreeBodiesSpeciesEnvironment {
            cutoff: cutoff,
            self_contribution: false,
        }
    }

    /// Create a new `ThreeBodiesSpeciesEnvironment` with the given `cutoff`, including
    /// self contributions.
    pub fn with_self_contribution(cutoff: f64) -> ThreeBodiesSpeciesEnvironment {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for ThreeBodiesSpeciesEnvironment");
        ThreeBodiesSpeciesEnvironment {
            cutoff: cutoff,
            self_contribution: true,
        }
    }
}

impl EnvironmentIndexes for ThreeBodiesSpeciesEnvironment {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]
    }

    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes {
        // Accumulate indexes in a set first to ensure uniqueness of the indexes
        // even if their are multiple neighbors of the same specie around a
        // given center
        let mut set = BTreeSet::new();

        let sort_pair = |i, j| {
            if i < j { (i, j) } else { (j, i) }
        };
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            let species = system.species();

            for center in 0..system.size() {
                for (i, j) in triplets_around(*system, center) {
                    let (species_1, species_2) = sort_pair(species[i], species[j]);
                    set.insert((i_system, center, species[center], species_1, species_2));
                }
            }

            if self.self_contribution {
                for (center, &species) in species.iter().enumerate() {
                    set.insert((i_system, center, species, species, species));
                }
            }
        }

        let mut indexes = IndexesBuilder::new(self.names());
        for (structure, center, species_center, species_1, species_2) in set {
            indexes.add(&[
                IndexValue::from(structure),
                IndexValue::from(center),
                IndexValue::from(species_center),
                IndexValue::from(species_1),
                IndexValue::from(species_2)
            ]);
        }
        return indexes.finish();
    }

    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), self.names());

        let sort_pair = |i, j| {
            if i < j { (i, j) } else { (j, i) }
        };

        // We need IndexSet to yield the indexes in the right order, i.e. the
        // order corresponding to whatever was passed in `samples`
        let mut indexes = IndexSet::new();
        for requested in samples {
            let i_system = requested[0];
            let center = requested[1].usize();
            let species_neighbor_1 = requested[3].usize();
            let species_neighbor_2 = requested[4].usize();

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff);

            let species = system.species();

            // FIXME: this will always be 0, but is required for Descriptor.densify
            if self.self_contribution && species[center] == species_neighbor_1 && species[center] == species_neighbor_2 {
                indexes.insert((i_system, center, species_neighbor_1, species_neighbor_1, species_neighbor_1, center));
            }

            for (i, j) in triplets_around(&*system, center) {
                let (species_1, species_2) = sort_pair(species[i], species[j]);
                indexes.insert((i_system, center, species[center], species_1, species_2, i));
                indexes.insert((i_system, center, species[center], species_1, species_2, j));
            }
        }

        let mut gradients = IndexesBuilder::new(vec![
            "structure", "center", "species_center", "species_neighbor_1",
            "species_neighbor_2", "neighbor", "spatial"
        ]);
        for (system, center, species_center, species_neighbor_1, species_neighbor_2, neighbor) in indexes {
            let center = IndexValue::from(center);
            let species_center = IndexValue::from(species_center);
            let species_neighbor_1 = IndexValue::from(species_neighbor_1);
            let species_neighbor_2 = IndexValue::from(species_neighbor_2);
            let neighbor = IndexValue::from(neighbor);
            for spatial in 0..3_usize {
                gradients.add(&[
                    system, center, species_center, species_neighbor_1,
                    species_neighbor_2, neighbor, IndexValue::from(spatial)
                ]);
            }
        }

        return Some(gradients.finish());
    }
}

/// Build the list of triplet i-center-j
fn triplets_around<'a>(system: &'a dyn System, center: usize) -> impl Iterator<Item=(usize, usize)> + 'a {
    let pairs = system.pairs_containing(center);

    return pairs.iter().cartesian_product(pairs).map(move |(first_pair, second_pair)| {
        let i = if first_pair.first == center {
            first_pair.second
        } else {
            first_pair.first
        };

        let j = if second_pair.first == center {
            second_pair.second
        } else {
            second_pair.first
        };

        return (i, j);
    });
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_systems;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::indexes::IndexValue::from($value as f64)
        };
    }

    #[test]
    fn structure() {
        let mut systems = test_systems(&["methane", "methane", "water"]);
        let indexes = StructureSpeciesEnvironment.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 6);
        assert_eq!(indexes.names(), &["structure", "species"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[v!(0), v!(1)], &[v!(0), v!(6)],
            &[v!(1), v!(1)], &[v!(1), v!(6)],
            &[v!(2), v!(1)], &[v!(2), v!(123456)],
        ]);
    }

    #[test]
    fn structure_gradient() {
        let mut systems = test_systems(&["CH", "water"]);
        let (_, gradients) = StructureSpeciesEnvironment.with_gradients(&mut systems.get());
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 15);
        assert_eq!(gradients.names(), &["structure", "species", "atom", "spatial"]);

        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel in CH
            &[v!(0), v!(1), v!(0), v!(0)], &[v!(0), v!(1), v!(0), v!(1)], &[v!(0), v!(1), v!(0), v!(2)],
            // C channel in CH
            &[v!(0), v!(6), v!(1), v!(0)], &[v!(0), v!(6), v!(1), v!(1)], &[v!(0), v!(6), v!(1), v!(2)],
            // H channel in water
            &[v!(1), v!(1), v!(1), v!(0)], &[v!(1), v!(1), v!(1), v!(1)], &[v!(1), v!(1), v!(1), v!(2)],
            &[v!(1), v!(1), v!(2), v!(0)], &[v!(1), v!(1), v!(2), v!(1)], &[v!(1), v!(1), v!(2), v!(2)],
            // O channel in water
            &[v!(1), v!(123456), v!(0), v!(0)], &[v!(1), v!(123456), v!(0), v!(1)], &[v!(1), v!(123456), v!(0), v!(2)],
        ]);
    }

    #[test]
    fn partial_structure_gradient() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "species"]);
        indexes.add(&[v!(2), v!(1)]);
        indexes.add(&[v!(0), v!(6)]);

        let mut systems = test_systems(&["CH", "water", "CH"]);
        let gradients = StructureSpeciesEnvironment.gradients_for(&mut systems.get(), &indexes.finish());
        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["structure", "species", "atom", "spatial"]);

        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel in CH #2
            &[v!(2), v!(1), v!(0), v!(0)],
            &[v!(2), v!(1), v!(0), v!(1)],
            &[v!(2), v!(1), v!(0), v!(2)],
            // C channel in CH #1
            &[v!(0), v!(6), v!(1), v!(0)],
            &[v!(0), v!(6), v!(1), v!(1)],
            &[v!(0), v!(6), v!(1), v!(2)],
        ]);
    }

    #[test]
    fn atoms() {
        let mut systems = test_systems(&["CH", "water"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 7);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H in CH
            &[v!(0), v!(0), v!(1), v!(6)],
            // C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
            // O in water
            &[v!(1), v!(0), v!(123456), v!(1)],
            // first H in water
            &[v!(1), v!(1), v!(1), v!(1)],
            &[v!(1), v!(1), v!(1), v!(123456)],
            // second H in water
            &[v!(1), v!(2), v!(1), v!(1)],
            &[v!(1), v!(2), v!(1), v!(123456)],
        ]);
    }

    #[test]
    fn atoms_self_contribution() {
        let mut systems = test_systems(&["CH"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 2);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H in CH
            &[v!(0), v!(0), v!(1), v!(6)],
            // C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
        ]);

        let strategy = AtomSpeciesEnvironment::with_self_contribution(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 4);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H in CH
            &[v!(0), v!(0), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6)],
            // C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
            &[v!(0), v!(1), v!(6), v!(6)],
        ]);

        // we get entries even without proper neighbors
        let strategy = AtomSpeciesEnvironment::with_self_contribution(1.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 2);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H in CH
            &[v!(0), v!(0), v!(1), v!(1)],
            // C in CH
            &[v!(0), v!(1), v!(6), v!(6)],
        ]);
    }

    #[test]
    fn atoms_gradient() {
        let mut systems = test_systems(&["CH", "water"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let (_, gradients) = strategy.with_gradients(&mut systems.get());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-C channel in CH
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // C-H channel in CH
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(0)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(1)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(2)],
            // O-H channel in water
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(2)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(2)],
            // H-H channel in water, 1st atom
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(2)],
            // H-O channel in water, 1st atom
            &[v!(1), v!(1), v!(1), v!(123456), v!(0), v!(0)],
            &[v!(1), v!(1), v!(1), v!(123456), v!(0), v!(1)],
            &[v!(1), v!(1), v!(1), v!(123456), v!(0), v!(2)],
            // H-H channel in water, 2nd atom
            &[v!(1), v!(2), v!(1), v!(1), v!(1), v!(0)],
            &[v!(1), v!(2), v!(1), v!(1), v!(1), v!(1)],
            &[v!(1), v!(2), v!(1), v!(1), v!(1), v!(2)],
            // H-O channel in water, 2nd atom
            &[v!(1), v!(2), v!(1), v!(123456), v!(0), v!(0)],
            &[v!(1), v!(2), v!(1), v!(123456), v!(0), v!(1)],
            &[v!(1), v!(2), v!(1), v!(123456), v!(0), v!(2)],
        ]);
    }

    #[test]
    fn partial_atoms_gradient() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        indexes.add(&[v!(1), v!(0), v!(123456), v!(1)]);
        indexes.add(&[v!(0), v!(0), v!(1), v!(6)]);
        indexes.add(&[v!(1), v!(1), v!(1), v!(1)]);

        let mut systems = test_systems(&["CH", "water"]);
        let strategy = AtomSpeciesEnvironment::new(2.0);
        let gradients = strategy.gradients_for(&mut systems.get(), &indexes.finish());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // O-H channel in water
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(2)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(2)],
            // H-C channel in CH
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // H-H channel in water, 1st atom
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(2)],
        ]);
    }

    #[test]
    fn three_bodies() {
        let mut systems = test_systems(&["CH", "water"]);
        let strategy = ThreeBodiesSpeciesEnvironment::new(2.0);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 9);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // C-H-C in CH
            &[v!(0), v!(0), v!(1), v!(6), v!(6)],
            // H-C-H in CH
            &[v!(0), v!(1), v!(6), v!(1), v!(1)],
            // H-O-H in water
            &[v!(1), v!(0), v!(123456), v!(1), v!(1)],
            // first H in water
            // H-H-H
            &[v!(1), v!(1), v!(1), v!(1), v!(1)],
            // H-H-O / O-H-H
            &[v!(1), v!(1), v!(1), v!(1), v!(123456)],
            // O-H-O
            &[v!(1), v!(1), v!(1), v!(123456), v!(123456)],
            // second H in water
            // H-H-H
            &[v!(1), v!(2), v!(1), v!(1), v!(1)],
            // H-H-O / O-H-H
            &[v!(1), v!(2), v!(1), v!(1), v!(123456)],
            // O-H-O
            &[v!(1), v!(2), v!(1), v!(123456), v!(123456)],
        ]);
    }

    #[test]
    fn three_bodies_self_contribution() {
        let mut systems = test_systems(&["water"]);
        // Only include O-H neighbors
        let strategy = ThreeBodiesSpeciesEnvironment::with_self_contribution(1.2);
        let indexes = strategy.indexes(&mut systems.get());
        assert_eq!(indexes.count(), 6);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H-O-H
            &[v!(0), v!(0), v!(123456), v!(1), v!(1)],
            // O-O-O
            &[v!(0), v!(0), v!(123456), v!(123456), v!(123456)],
            // first H in water
            // H-H-H
            &[v!(0), v!(1), v!(1), v!(1), v!(1)],
            // O-H-O
            &[v!(0), v!(1), v!(1), v!(123456), v!(123456)],
            // second H in water
            // H-H-H
            &[v!(0), v!(2), v!(1), v!(1), v!(1)],
            // O-H-O
            &[v!(0), v!(2), v!(1), v!(123456), v!(123456)],
        ]);
    }

    #[test]
    fn three_bodies_gradients() {
        let mut systems = test_systems(&["water"]);
        let strategy = ThreeBodiesSpeciesEnvironment::new(2.0);
        let (_, gradients) = strategy.with_gradients(&mut systems.get());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 30);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-O-H in water
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(1), v!(0)],
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(1), v!(1)],
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(1), v!(2)],
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(2), v!(0)],
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(2), v!(1)],
            &[v!(0), v!(0), v!(123456), v!(1), v!(1), v!(2), v!(2)],
            // O-H-O, 1rst H
            &[v!(0), v!(1), v!(1), v!(123456), v!(123456), v!(0), v!(0)],
            &[v!(0), v!(1), v!(1), v!(123456), v!(123456), v!(0), v!(1)],
            &[v!(0), v!(1), v!(1), v!(123456), v!(123456), v!(0), v!(2)],
            // H-H-O, 1rst H
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(0), v!(0)],
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(0), v!(1)],
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(0), v!(2)],
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(2), v!(0)],
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(2), v!(1)],
            &[v!(0), v!(1), v!(1), v!(1), v!(123456), v!(2), v!(2)],
            // H-H-H 1rst H
            &[v!(0), v!(1), v!(1), v!(1), v!(1), v!(2), v!(0)],
            &[v!(0), v!(1), v!(1), v!(1), v!(1), v!(2), v!(1)],
            &[v!(0), v!(1), v!(1), v!(1), v!(1), v!(2), v!(2)],
            // O-H-O, 2nd H
            &[v!(0), v!(2), v!(1), v!(123456), v!(123456), v!(0), v!(0)],
            &[v!(0), v!(2), v!(1), v!(123456), v!(123456), v!(0), v!(1)],
            &[v!(0), v!(2), v!(1), v!(123456), v!(123456), v!(0), v!(2)],
            // H-H-O, 2nd H
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(0), v!(0)],
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(0), v!(1)],
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(0), v!(2)],
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(1), v!(0)],
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(1), v!(1)],
            &[v!(0), v!(2), v!(1), v!(1), v!(123456), v!(1), v!(2)],
            // H-H-H 2nd H
            &[v!(0), v!(2), v!(1), v!(1), v!(1), v!(1), v!(0)],
            &[v!(0), v!(2), v!(1), v!(1), v!(1), v!(1), v!(1)],
            &[v!(0), v!(2), v!(1), v!(1), v!(1), v!(1), v!(2)]
        ]);
    }
}
