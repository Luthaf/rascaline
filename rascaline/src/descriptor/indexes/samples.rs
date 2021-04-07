use indexmap::IndexSet;

use crate::systems::System;
use super::{Indexes, IndexesBuilder, SamplesIndexes, IndexValue};

/// `StructureSamples` is used to represents samples corresponding to full
/// structures, each structure being described by a single features vector.
///
/// It does not contain any chemical species information, for this you should
/// use [`super::StructureSpeciesSamples`].
///
/// The base set of indexes contains only the `structure` index; the  gradient
/// indexes also contains the `atom` inside the structure with respect to which
/// the gradient is taken and the `spatial` (i.e. x/y/z) index.
pub struct StructureSamples;

impl SamplesIndexes for StructureSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure"]
    }

    #[time_graph::instrument(name = "StructureSamples::indexes")]
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut indexes = IndexesBuilder::new(self.names());
        for system in 0..systems.len() {
            indexes.add(&[IndexValue::from(system)]);
        }
        return indexes.finish();
    }

    #[time_graph::instrument(name = "StructureSamples::gradients_for")]
    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), self.names());

        let mut gradients = IndexesBuilder::new(vec!["structure", "atom", "spatial"]);
        for value in samples.iter() {
            let system = value[0];
            for atom in 0..systems[system.usize()].size() {
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(0)]);
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(1)]);
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(2)]);
            }
        }

        Some(gradients.finish())
    }
}

/// `AtomSamples` is used to represents atom-centered environments, where
/// each atom in a structure is described with a feature vector based on other
/// atoms inside a sphere centered on the central atom.
///
/// This type of indexes does not contain any chemical species information, for
/// this you should use [`super::AtomSpeciesSamples`].
///
/// The base set of indexes contains `structure` and `center` (i.e. central atom
/// index inside the structure); the gradient indexes also contains the
/// `neighbor` inside the spherical cutoff with respect to which the gradient is
/// taken and the `spatial` (i.e x/y/z) index.
pub struct AtomSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
}

impl AtomSamples {
    /// Create a new `AtomSamples` with the given cutoff.
    pub fn new(cutoff: f64) -> AtomSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for AtomSamples");
        AtomSamples { cutoff }
    }
}

impl SamplesIndexes for AtomSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center"]
    }

    #[time_graph::instrument(name = "AtomSamples::indexes")]
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut indexes = IndexesBuilder::new(self.names());
        for (i_system, system) in systems.iter().enumerate() {
            for center in 0..system.size() {
                indexes.add(&[IndexValue::from(i_system), IndexValue::from(center)]);
            }
        }
        return indexes.finish();
    }

    #[time_graph::instrument(name = "AtomSamples::gradients_for")]
    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Option<Indexes> {
        assert_eq!(samples.names(), self.names());

        // We need IndexSet to yield the indexes in the right order, i.e. the
        // order corresponding to whatever was passed in sample
        let mut indexes = IndexSet::new();
        for requested in samples {
            let i_system = requested[0];
            let center = requested[1].usize();
            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff);

            for pair in system.pairs_containing(center) {
                if pair.first == center {
                    indexes.insert((i_system, pair.first, pair.second));
                } else if pair.second == center {
                    indexes.insert((i_system, pair.second, pair.first));
                }
            }
        }

        let mut gradients = IndexesBuilder::new(vec!["structure", "center", "neighbor", "spatial"]);
        for (structure, atom, neighbor) in indexes {
            let atom = IndexValue::from(atom);
            let neighbor = IndexValue::from(neighbor);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(0)]);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(1)]);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(2)]);
        }

        return Some(gradients.finish());
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_systems;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::indexes::IndexValue::from($value)
        };
    }

    #[test]
    fn structure() {
        let mut systems = test_systems(&["methane", "methane", "water"]).boxed();
        let indexes = StructureSamples.indexes(&mut systems);
        assert_eq!(indexes.count(), 3);
        assert_eq!(indexes.names(), &["structure"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![&[v!(0)], &[v!(1)], &[v!(2)]]);
    }

    #[test]
    fn structure_gradient() {
        let mut systems = test_systems(&["methane", "water"]).boxed();

        let (_, gradients) = StructureSamples.with_gradients(&mut systems);
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // methane
            &[v!(0), v!(0), v!(0)], &[v!(0), v!(0), v!(1)], &[v!(0), v!(0), v!(2)],
            &[v!(0), v!(1), v!(0)], &[v!(0), v!(1), v!(1)], &[v!(0), v!(1), v!(2)],
            &[v!(0), v!(2), v!(0)], &[v!(0), v!(2), v!(1)], &[v!(0), v!(2), v!(2)],
            &[v!(0), v!(3), v!(0)], &[v!(0), v!(3), v!(1)], &[v!(0), v!(3), v!(2)],
            &[v!(0), v!(4), v!(0)], &[v!(0), v!(4), v!(1)], &[v!(0), v!(4), v!(2)],
            // water
            &[v!(1), v!(0), v!(0)], &[v!(1), v!(0), v!(1)], &[v!(1), v!(0), v!(2)],
            &[v!(1), v!(1), v!(0)], &[v!(1), v!(1), v!(1)], &[v!(1), v!(1), v!(2)],
            &[v!(1), v!(2), v!(0)], &[v!(1), v!(2), v!(1)], &[v!(1), v!(2), v!(2)],
        ]);
    }

    #[test]
    fn partial_structure_gradient() {
        let mut indexes = IndexesBuilder::new(vec!["structure"]);
        indexes.add(&[v!(2)]);
        indexes.add(&[v!(0)]);

        let mut systems = test_systems(&["water", "methane", "water", "methane"]).boxed();
        let gradients = StructureSamples.gradients_for(&mut systems, &indexes.finish());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["structure", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // water #2
            &[v!(2), v!(0), v!(0)], &[v!(2), v!(0), v!(1)], &[v!(2), v!(0), v!(2)],
            &[v!(2), v!(1), v!(0)], &[v!(2), v!(1), v!(1)], &[v!(2), v!(1), v!(2)],
            &[v!(2), v!(2), v!(0)], &[v!(2), v!(2), v!(1)], &[v!(2), v!(2), v!(2)],
            // water #1
            &[v!(0), v!(0), v!(0)], &[v!(0), v!(0), v!(1)], &[v!(0), v!(0), v!(2)],
            &[v!(0), v!(1), v!(0)], &[v!(0), v!(1), v!(1)], &[v!(0), v!(1), v!(2)],
            &[v!(0), v!(2), v!(0)], &[v!(0), v!(2), v!(1)], &[v!(0), v!(2), v!(2)],
        ]);
    }

    #[test]
    fn atoms() {
        let mut systems = test_systems(&["methane", "water"]).boxed();
        let strategy = AtomSamples { cutoff: 2.0 };
        let indexes = strategy.indexes(&mut systems);
        assert_eq!(indexes.count(), 8);
        assert_eq!(indexes.names(), &["structure", "center"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[v!(0), v!(0)], &[v!(0), v!(1)], &[v!(0), v!(2)], &[v!(0), v!(3)], &[v!(0), v!(4)],
            &[v!(1), v!(0)], &[v!(1), v!(1)], &[v!(1), v!(2)],
        ]);
    }

    #[test]
    fn atom_gradients() {
        let mut systems = test_systems(&["methane"]).boxed();
        let strategy = AtomSamples { cutoff: 1.5 };
        let (_, gradients) = strategy.with_gradients(&mut systems);
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "center", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // Only C-H neighbors are within 1.3 A
            // C center
            &[v!(0), v!(0), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(2)],

            &[v!(0), v!(0), v!(2), v!(0)],
            &[v!(0), v!(0), v!(2), v!(1)],
            &[v!(0), v!(0), v!(2), v!(2)],

            &[v!(0), v!(0), v!(3), v!(0)],
            &[v!(0), v!(0), v!(3), v!(1)],
            &[v!(0), v!(0), v!(3), v!(2)],

            &[v!(0), v!(0), v!(4), v!(0)],
            &[v!(0), v!(0), v!(4), v!(1)],
            &[v!(0), v!(0), v!(4), v!(2)],
            // H centers
            &[v!(0), v!(1), v!(0), v!(0)],
            &[v!(0), v!(1), v!(0), v!(1)],
            &[v!(0), v!(1), v!(0), v!(2)],

            &[v!(0), v!(2), v!(0), v!(0)],
            &[v!(0), v!(2), v!(0), v!(1)],
            &[v!(0), v!(2), v!(0), v!(2)],

            &[v!(0), v!(3), v!(0), v!(0)],
            &[v!(0), v!(3), v!(0), v!(1)],
            &[v!(0), v!(3), v!(0), v!(2)],

            &[v!(0), v!(4), v!(0), v!(0)],
            &[v!(0), v!(4), v!(0), v!(1)],
            &[v!(0), v!(4), v!(0), v!(2)],
        ]);
    }

    #[test]
    fn partial_atom_gradient() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "center"]);
        // out of order values to ensure the gradients are also out of order
        indexes.add(&[v!(0), v!(2)]);
        indexes.add(&[v!(0), v!(0)]);

        let mut systems = test_systems(&["methane"]).boxed();
        let strategy = AtomSamples { cutoff: 1.5 };
        let gradients = strategy.gradients_for(&mut systems, &indexes.finish());
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["structure", "center", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H centers
            &[v!(0), v!(2), v!(0), v!(0)],
            &[v!(0), v!(2), v!(0), v!(1)],
            &[v!(0), v!(2), v!(0), v!(2)],
            // C center
            &[v!(0), v!(0), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(2)],

            &[v!(0), v!(0), v!(2), v!(0)],
            &[v!(0), v!(0), v!(2), v!(1)],
            &[v!(0), v!(0), v!(2), v!(2)],

            &[v!(0), v!(0), v!(3), v!(0)],
            &[v!(0), v!(0), v!(3), v!(1)],
            &[v!(0), v!(0), v!(3), v!(2)],

            &[v!(0), v!(0), v!(4), v!(0)],
            &[v!(0), v!(0), v!(4), v!(1)],
            &[v!(0), v!(0), v!(4), v!(2)],
        ]);
    }
}
