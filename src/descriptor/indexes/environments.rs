use std::collections::BTreeSet;

use crate::system::System;
use super::{Indexes, IndexesBuilder, EnvironmentIndexes};

pub struct StructureIdx;

impl EnvironmentIndexes for StructureIdx {
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut indexes = IndexesBuilder::new(vec!["structure"]);
        for system in 0..systems.len() {
            indexes.add(&[system]);
        }
        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
        let mut gradients = IndexesBuilder::new(vec!["structure", "atom", "spatial"]);
        for system in 0..systems.len() {
            for atom in 0..systems[system].size() {
                gradients.add(&[system, atom, 0]);
                gradients.add(&[system, atom, 1]);
                gradients.add(&[system, atom, 2]);
            }
        }
        return (self.indexes(systems), Some(gradients.finish()));
    }
}

pub struct AtomIdx {
    cutoff: f64,
}

impl AtomIdx {
    pub fn new(cutoff: f64) -> AtomIdx {
        assert!(cutoff > 0.0, "cutoff must be positive for AtomIdx");
        AtomIdx {
            cutoff: cutoff
        }
    }
}

impl EnvironmentIndexes for AtomIdx {
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes {
        let mut indexes = IndexesBuilder::new(vec!["structure", "atom"]);
        for system in 0..systems.len() {
            for atom in 0..systems[system].size() {
                indexes.add(&[system, atom]);
            }
        }
        return indexes.finish();
    }

    fn with_gradients(&self, systems: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
        // a BTreeSet will yield the indexes in the right order
        let mut indexes = BTreeSet::new();
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff);
            system.neighbors().foreach_pair(&mut |i, j, _| {
                indexes.insert((i_system, i, j));
                indexes.insert((i_system, j, i));
            })
        }

        let mut gradients = IndexesBuilder::new(vec!["structure", "atom", "neighbor", "spatial"]);
        for (structure, atom, neighbor) in indexes {
            gradients.add(&[structure, atom, neighbor, 0]);
            gradients.add(&[structure, atom, neighbor, 1]);
            gradients.add(&[structure, atom, neighbor, 2]);
        }

        return (self.indexes(systems), Some(gradients.finish()));
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_systems;

    #[test]
    fn structure() {
        let systems = &mut test_systems(vec!["methane", "methane", "water"]);
        let indexes = StructureIdx.indexes(systems);
        assert_eq!(indexes.count(), 3);
        assert_eq!(indexes.names(), &["structure"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![&[0], &[1], &[2]]);
    }

    #[test]
    fn structure_gradient() {
        let systems = &mut test_systems(vec!["methane", "water"]);

        let (_, gradients) = StructureIdx.with_gradients(systems);
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // methane
            &[0, 0, 0], &[0, 0, 1], &[0, 0, 2],
            &[0, 1, 0], &[0, 1, 1], &[0, 1, 2],
            &[0, 2, 0], &[0, 2, 1], &[0, 2, 2],
            &[0, 3, 0], &[0, 3, 1], &[0, 3, 2],
            &[0, 4, 0], &[0, 4, 1], &[0, 4, 2],
            // water
            &[1, 0, 0], &[1, 0, 1], &[1, 0, 2],
            &[1, 1, 0], &[1, 1, 1], &[1, 1, 2],
            &[1, 2, 0], &[1, 2, 1], &[1, 2, 2],
        ]);
    }


    #[test]
    fn atoms() {
        let systems = &mut test_systems(vec!["methane", "water"]);
        let strategy = AtomIdx { cutoff: 2.0 };
        let indexes = strategy.indexes(systems);
        assert_eq!(indexes.count(), 8);
        assert_eq!(indexes.names(), &["structure", "atom"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[0, 0], &[0, 1], &[0, 2], &[0, 3], &[0, 4],
            &[1, 0], &[1, 1], &[1, 2],
        ]);
    }

    #[test]
    fn atom_gradients() {
        let systems = &mut test_systems(vec!["methane"]);
        let strategy = AtomIdx { cutoff: 1.5 };
        let (_, gradients) = strategy.with_gradients(systems);
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "atom", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // Only C-H neighbors are within 1.3 A
            // C center
            &[0, 0, 1, 0], &[0, 0, 1, 1], &[0, 0, 1, 2],
            &[0, 0, 2, 0], &[0, 0, 2, 1], &[0, 0, 2, 2],
            &[0, 0, 3, 0], &[0, 0, 3, 1], &[0, 0, 3, 2],
            &[0, 0, 4, 0], &[0, 0, 4, 1], &[0, 0, 4, 2],
            // H centers
            &[0, 1, 0, 0], &[0, 1, 0, 1], &[0, 1, 0, 2],
            &[0, 2, 0, 0], &[0, 2, 0, 1], &[0, 2, 0, 2],
            &[0, 3, 0, 0], &[0, 3, 0, 1], &[0, 3, 0, 2],
            &[0, 4, 0, 0], &[0, 4, 0, 1], &[0, 4, 0, 2],
        ]);
    }
}
