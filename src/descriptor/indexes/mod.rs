use std::ffi::CString;
use std::collections::BTreeSet;

use crate::system::System;

mod environments;
pub use self::environments::{StructureEnvironment, AtomEnvironment};

mod species;
pub use self::species::{StructureSpeciesEnvironment, AtomSpeciesEnvironment};

pub struct IndexesBuilder {
    /// Names of the indexes
    names: Vec<String>,
    /// Values of the indexes, as a linearized 2D array in row-major order
    values: Vec<usize>,
}

impl IndexesBuilder {
    /// Create a new empty `IndexesBuilder` with the given `names`
    pub fn new(names: Vec<&str>) -> IndexesBuilder {
        for name in &names {
            if !is_valid_ident(name) {
                panic!("all indexes names must be valid identifiers, '{}' is not", name);
            }
        }

        if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
            panic!("invalid indexes: the same name is used multiple times");
        }

        IndexesBuilder {
            names: names.into_iter().map(|s| s.into()).collect(),
            values: Vec::new(),
        }
    }

    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single entry with the given `values` for this set of indexes
    pub fn add(&mut self, values: &[usize]) {
        assert_eq!(
            self.size(), values.len(),
            "wrong size for added index: got {}, but expected {}", values.len(), self.size()
        );
        for chunk in self.values.chunks_exact(self.size()) {
            if chunk == values {
                panic!(
                    "can not have the same index value multiple time: [{}] is already present",
                    values.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
                );
            }
        }
        self.values.extend(values);
    }

    pub fn finish(self) -> Indexes {
        Indexes {
            names: self.names.into_iter()
                .map(|s| CString::new(s).expect("invalid C string"))
                .collect(),
            values: self.values,
        }
    }
}

fn is_valid_ident(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    for (i, c) in name.chars().enumerate() {
        if i == 0 {
            if c.is_ascii_digit() {
                return false;
            }
        }

        if !(c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
    }

    return true;
}

#[derive(Clone, Debug)]
pub struct Indexes {
    /// Names of the indexes, stored as C strings for easier integration
    /// with the C API
    names: Vec<CString>,
    /// Values of the indexes, as a linearized 2D array in row-major order
    values: Vec<usize>,
}

impl Indexes {
    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Names of the indexes
    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.to_str().expect("invalid UTF8")).collect()
    }

    /// Names of the indexes as C-compatible (null terminated) strings
    pub fn c_names(&self) -> &[CString] {
        &self.names
    }

    /// How many entries of indexes do we have
    pub fn count(&self) -> usize {
        if self.size() == 0 {
            return 0;
        } else {
            return self.values.len() / self.size();
        }
    }

    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            size: self.names.len(),
            values: &self.values
        };
    }

    pub fn contains(&self, value: &[usize]) -> bool {
        if value.len() != self.size() {
            return false;
        }

        for v in self.iter() {
            if v == value {
                return true;
            }
        }

        return false;
    }
}

pub struct Iter<'a> {
    size: usize,
    values: &'a [usize],
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a[usize];
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.len() == 0 {
            return None
        } else {
            let (value, rest) = self.values.split_at(self.size);
            self.values = rest;
            return Some(value);
        }
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    fn len(&self) -> usize {
        self.values.len() / self.size
    }
}

impl<'a> IntoIterator for &'a Indexes {
    type IntoIter = Iter<'a>;
    type Item = &'a [usize];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for Indexes {
    type Output = [usize];
    fn index(&self, i: usize) -> &[usize] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}

pub trait EnvironmentIndexes {
    fn indexes(&self, systems: &mut [&mut dyn System]) -> Indexes;

    fn with_gradients(&self, systems: &mut [&mut dyn System]) -> (Indexes, Option<Indexes>) {
        let indexes = self.indexes(systems);
        let gradients = self.gradients_for(systems, &indexes);
        return (indexes, gradients);
    }

    #[allow(unused_variables)]
    fn gradients_for(&self, systems: &mut [&mut dyn System], samples: &Indexes) -> Option<Indexes> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes() {
        let mut builder = IndexesBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[4, 3]);

        let idx = builder.finish();
        assert_eq!(idx.names(), &["foo", "bar"]);
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.count(), 3);

        assert_eq!(idx[0], [2, 3]);
        assert_eq!(idx[1], [1, 2]);
        assert_eq!(idx[2], [4, 3]);
    }

    #[test]
    fn indexes_iter() {
        let mut builder = IndexesBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[4, 3]);

        let idx = builder.finish();
        let mut iter = idx.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next().unwrap(), &[1, 2]);
        assert_eq!(iter.next().unwrap(), &[4, 3]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "all indexes names must be valid identifiers, \'33 bar\' is not")]
    fn invalid_index_name() {
        let _ = IndexesBuilder::new(vec!["foo", "33 bar"]);
    }

    #[test]
    #[should_panic(expected = "invalid indexes: the same name is used multiple times")]
    fn duplicated_index_name() {
        let _ = IndexesBuilder::new(vec!["foo", "bar", "foo"]);
    }

    #[test]
    #[should_panic(expected = "can not have the same index value multiple time: [0, 1] is already present")]
    fn duplicated_index_value() {
        let mut builder = IndexesBuilder::new(vec!["foo", "bar"]);
        builder.add(&[0, 1]);
        builder.add(&[0, 1]);
    }
}
