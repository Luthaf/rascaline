use crate::system::System;

mod environments;
pub use self::environments::{StructureIdx, AtomIdx};

mod species;
pub use self::species::{StructureSpeciesIdx, PairSpeciesIdx};

pub struct IndexesBuilder {
    /// Names of the indexes
    names: Vec<&'static str>,
    /// Values of the indexes, as a linearized 2D array
    values: Vec<usize>,
}

impl IndexesBuilder {
    /// Create a new empty `IndexesBuilder` with the given `names`
    pub fn new(names: Vec<&'static str>) -> IndexesBuilder {
        IndexesBuilder {
            names: names,
            values: Vec::new(),
        }
    }

    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single entry with the given `values` for this set of indexes
    pub fn add(&mut self, values: &[usize]) {
        assert_eq!(self.size(), values.len());
        self.values.extend(values);
    }

    pub fn finish(self) -> Indexes {
        Indexes {
            names: self.names,
            values: self.values,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Indexes {
    /// Names of the indexes
    names: Vec<&'static str>,
    /// Values of the indexes, as a linearized 2D array
    values: Vec<usize>,
}

impl Indexes {
    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Names of the indexes
    pub fn names(&self) -> &[&str] {
        &self.names
    }

    /// How many entries of indexes do we have
    pub fn count(&self) -> usize {
        self.values.len() / self.size()
    }

    /// Get the value of the indexes at the given `linear` index
    pub fn value(&self, linear: usize) -> &[usize] {
        let start = linear * self.size();
        let stop = (linear + 1) * self.size();
        &self.values[start..stop]
    }

    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            size: self.names.len(),
            values: &self.values
        };
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

pub trait EnvironmentIndexes {
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Indexes;

    fn with_gradients(&self, systems: &mut [Box<dyn System>]) -> (Indexes, Option<Indexes>) {
        (self.indexes(systems), None)
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
        builder.add(&[2, 3]);

        let idx = builder.finish();
        assert_eq!(idx.names(), &["foo", "bar"]);
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.count(), 3);

        assert_eq!(idx.value(0), &[2, 3]);
        assert_eq!(idx.value(1), &[1, 2]);
        assert_eq!(idx.value(2), &[2, 3]);
    }

    #[test]
    fn indexes_iter() {
        let mut builder = IndexesBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[2, 3]);

        let idx = builder.finish();
        let mut iter = idx.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next().unwrap(), &[1, 2]);
        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next(), None);
    }
}
