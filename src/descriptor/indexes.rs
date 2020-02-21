use std::ops::{Deref, DerefMut};

use crate::system::System;

pub trait Indexes {
    /// Get the names of each index
    fn names(&self) -> &[&'static str];
    /// Initialize the set of indexes to work with the given systems
    fn initialize(&mut self, systems: &[&dyn System]);
    /// Get the total number of indexes
    fn count(&self) -> usize;
    /// Get the indexes corresponding to the gradient of the current indexes
    /// with respect to the position of another atom
    fn gradient(&self) -> Option<Box<dyn Indexes>>;
    /// Get the value of the indexes at the given `linear` index
    fn value(&self, linear: usize) -> &[usize];
}

impl Indexes for Box<dyn Indexes> {
    fn names(&self) -> &[&'static str] {
        self.deref().names()
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.deref_mut().initialize(systems)
    }

    fn count(&self) -> usize {
        self.deref().count()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        self.deref().gradient()
    }

    fn value(&self, linear: usize) -> &[usize] {
        self.deref().value(linear)
    }
}

struct CombinedIndexes<I, J> where I: Indexes, J: Indexes {
    first: I,
    second: J,
    names: Vec<&'static str>,
}

impl<I, J> CombinedIndexes<I, J> where I: Indexes, J: Indexes {
    pub fn new(first: I, second: J) -> CombinedIndexes<I, J> {
        let names = first.names()
                         .iter()
                         .chain(second.names())
                         .cloned()
                         .collect();
        return CombinedIndexes { first, second, names };
    }
}

impl<I, J> Indexes for CombinedIndexes<I, J> where I: Indexes, J: Indexes {
    fn names(&self) -> &[&'static str] {
        &self.names
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.first.initialize(systems);
        self.second.initialize(systems);
    }

    fn count(&self) -> usize {
        self.first.count() * self.second.count()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        let first = self.first.gradient();
        let second = self.second.gradient();
        match (first, second) {
            (None, _) => None,
            (_, None) => None,
            (Some(first), Some(second)) => Some(Box::new(CombinedIndexes::new(first, second)))
        }
    }

    fn value(&self, linear: usize) -> &[usize] {
        todo!()
    }
}

pub struct StructureIndexes {
    indexes: Vec<usize>,
}

impl StructureIndexes {
    pub fn new() -> StructureIndexes {
        StructureIndexes {
            indexes: Vec::new(),
        }
    }
}

impl Indexes for StructureIndexes {
    fn names(&self) -> &[&'static str] {
        &["structure"]
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.indexes.clear();
        for system in 0..systems.len() {
            self.indexes.push(system);
        }
    }

    fn count(&self) -> usize {
        self.indexes.len()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        Some(Box::new(AtomIndexes::new()))
    }

    fn value(&self, linear: usize) -> &[usize] {
        return std::slice::from_ref(&self.indexes[linear]);
    }
}

pub struct AtomIndexes {
    indexes: Vec<[usize; 2]>
}

impl AtomIndexes {
    pub fn new() -> AtomIndexes {
        AtomIndexes {
            indexes: Vec::new()
        }
    }
}

impl Indexes for AtomIndexes {
    fn names(&self) -> &[&'static str] {
        &["structure", "atom"]
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.indexes.clear();
        for (i_system, system) in systems.iter().enumerate() {
            for atom in 0..system.natoms() {
                self.indexes.push([i_system, atom]);
            }
        }
    }

    fn count(&self) -> usize {
        self.indexes.len()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        Some(Box::new(PairIndexes::new()))
    }

    fn value(&self, linear: usize) -> &[usize] {
        &self.indexes[linear]
    }
}

pub struct PairIndexes {
    indexes: Vec<[usize; 3]>
}

impl PairIndexes {
    pub fn new() -> PairIndexes {
        PairIndexes {
            indexes: Vec::new()
        }
    }
}

impl Indexes for PairIndexes {
    fn names(&self) -> &[&'static str] {
        &["structure", "first", "second"]
    }

    fn initialize(&mut self, systems: &[&dyn System]) {
        self.indexes.clear();
        for (i_system, system) in systems.iter().enumerate() {
            for first in 0..system.natoms() {
                for second in (first + 1)..system.natoms() {
                    self.indexes.push([i_system, first, second]);
                }
            }
        }
    }

    fn count(&self) -> usize {
        self.indexes.len()
    }

    fn gradient(&self) -> Option<Box<dyn Indexes>> {
        None
    }

    fn value(&self, linear: usize) -> &[usize] {
        &self.indexes[linear]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::test_system;

    #[test]
    fn structure() {
        let systems = [
            &*test_system("methane"),
            &*test_system("methane"),
            &*test_system("methane")
        ];

        let mut indexes = StructureIndexes::new();
        indexes.initialize(&systems);

        assert_eq!(indexes.count(), 3);
        assert_eq!(indexes.names(), &["structure"]);
        assert_eq!(indexes.value(0), &[0]);
        assert_eq!(indexes.value(1), &[1]);
        assert_eq!(indexes.value(2), &[2]);
    }

    #[test]
    fn structure_grad() {
        unimplemented!()
    }

    #[test]
    fn atom() {
        unimplemented!()
    }

    #[test]
    fn atom_grad() {
        unimplemented!()
    }

    #[test]
    fn pairs() {
        unimplemented!()
    }

    #[test]
    fn pairs_grad() {
        unimplemented!()
    }
}
