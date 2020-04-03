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
        let mut gradients = IndexesBuilder::new(vec!["spatial", "structure", "atom"]);

        for spatial in 0..3 {
            for system in 0..systems.len() {
                for atom in 0..systems[system].size() {
                    gradients.add(&[spatial, system, atom]);
                }
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

    fn with_gradients(&self, _: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
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
        assert_eq!(gradients.names(), &["spatial", "structure", "atom"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            &[0, 0, 0], &[0, 0, 1], &[0, 0, 2], &[0, 0, 3], &[0, 0, 4],
            &[0, 1, 0], &[0, 1, 1], &[0, 1, 2],
            &[1, 0, 0], &[1, 0, 1], &[1, 0, 2], &[1, 0, 3], &[1, 0, 4],
            &[1, 1, 0], &[1, 1, 1], &[1, 1, 2],
            &[2, 0, 0], &[2, 0, 1], &[2, 0, 2], &[2, 0, 3], &[2, 0, 4],
            &[2, 1, 0], &[2, 1, 1], &[2, 1, 2],
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
    #[ignore]
    fn atom_gradients() {
        todo!()
    }
}
