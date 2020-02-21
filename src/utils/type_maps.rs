use std::collections::{HashSet, HashMap};

use crate::system::System;

#[derive(Debug, Clone)]
pub struct AtomTypeMap {
    types: HashMap<usize, usize>,
}

impl Default for AtomTypeMap {
    fn default() -> AtomTypeMap {
        AtomTypeMap::new()
    }
}

impl AtomTypeMap {
    pub fn new() -> AtomTypeMap {
        AtomTypeMap {
            types: HashMap::new()
        }
    }

    pub fn initialize(&mut self, systems: &[&dyn System]) {
        let mut types = HashSet::new();
        for system in systems {
            for &atom_type in system.types() {
                types.insert(atom_type);
            }
        }

        self.types = types.iter().cloned().enumerate().collect();
    }

    pub fn validate(&self, system: &dyn System) -> bool {
        for atom_type in system.types() {
            if !self.types.contains_key(atom_type) {
                return false;
            }
        }
        return true;
    }

    pub fn types<'a>(&'a self) -> impl Iterator<Item=usize> + 'a {
        self.types.iter().map(|(k, _)| k).cloned()
    }

    pub fn get(&self, atom_type: usize) -> usize {
        self.types.get(&atom_type).cloned().expect("invalid atom_type in AtomTypeMap::get")
    }

    pub fn count(&self) -> usize {
        self.types.len()
    }
}
