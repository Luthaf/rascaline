use std::ffi::CString;
use std::collections::BTreeSet;

use crate::system::System;

/// Biggest integer value such that all integer values up to it can be stored in
/// a f64 value.
const MAX_SAFE_INTEGER: isize = 9007199254740991;

/// Smallest integer value such that all integers value between it and zero can
/// be stored in a f64 value.
const MIN_SAFE_INTEGER: isize = -9007199254740991;


#[derive(Clone, Copy, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct IndexValue(f64);

#[allow(clippy::derive_hash_xor_eq)]
impl std::hash::Hash for IndexValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_le_bytes().hash(state);
    }
}

// This is fine since we can not construct a NaN IndexValue
impl Ord for IndexValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).expect("a NaN slipped through!")
    }
}

impl PartialOrd for IndexValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl Eq for IndexValue {}

impl std::fmt::Debug for IndexValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for IndexValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<f64> for IndexValue {
    fn from(value: f64) -> IndexValue {
        assert!(!value.is_nan());
        IndexValue(value)
    }
}

impl From<usize> for IndexValue {
    fn from(value: usize) -> IndexValue {
        assert!(value < MAX_SAFE_INTEGER as usize);
        IndexValue(value as f64)
    }
}

impl From<isize> for IndexValue {
    fn from(value: isize) -> IndexValue {
        assert!(MIN_SAFE_INTEGER < value && value < MAX_SAFE_INTEGER);
        IndexValue(value as f64)
    }
}

impl IndexValue {
    pub fn f64(self) -> f64 {
        self.0
    }

    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0.0 && self.0 < MAX_SAFE_INTEGER as f64 && self.0 % 1.0 == 0.0);
        self.0 as usize
    }

    pub fn isize(self) -> isize {
        debug_assert!((MIN_SAFE_INTEGER as f64) < self.0 && self.0 < MAX_SAFE_INTEGER as f64 && self.0 % 1.0 == 0.0);
        self.0 as isize
    }
}

pub struct IndexesBuilder {
    /// Names of the indexes
    names: Vec<String>,
    /// Values of the indexes, as a linearized 2D array in row-major order
    values: Vec<IndexValue>,
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
    pub fn add(&mut self, values: &[IndexValue]) {
        assert_eq!(
            self.size(), values.len(),
            "wrong size for added index: got {}, but expected {}", values.len(), self.size()
        );

        for chunk in self.values.chunks_exact(self.size()) {
            if chunk == values {
                panic!(
                    "can not have the same index value multiple time: [{}] is already present",
                    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
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
        if i == 0 && c.is_ascii_digit() {
            return false;
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
    values: Vec<IndexValue>,
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

    /// Check whether the given `value` is part of this set of indexes
    pub fn contains(&self, value: &[IndexValue]) -> bool {
        self.position(value).is_some()
    }

    /// Get the position of the given value on this set of indexes, or None.
    pub fn position(&self, value: &[IndexValue]) -> Option<usize> {
        if value.len() != self.size() {
            return None;
        }

        for (i, v) in self.iter().enumerate() {
            if v == value {
                return Some(i);
            }
        }

        return None;
    }
}

pub struct Iter<'a> {
    size: usize,
    values: &'a [IndexValue],
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a[IndexValue];
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.is_empty() {
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
    type Item = &'a [IndexValue];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for Indexes {
    type Output = [IndexValue];
    fn index(&self, i: usize) -> &[IndexValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}

#[allow(clippy::module_name_repetitions)]
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
        builder.add(&[IndexValue::from(2_isize), IndexValue::from(3.9)]);
        builder.add(&[IndexValue::from(1_isize), IndexValue::from(2.43)]);
        builder.add(&[IndexValue::from(-4_isize), IndexValue::from(-24e13)]);

        let idx = builder.finish();
        assert_eq!(idx.names(), &["foo", "bar"]);
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.count(), 3);

        assert_eq!(idx[0], [IndexValue::from(2_isize), IndexValue::from(3.9)]);
        assert_eq!(idx[1], [IndexValue::from(1_isize), IndexValue::from(2.43)]);
        assert_eq!(idx[2], [IndexValue::from(-4_isize), IndexValue::from(-24e13)]);
    }

    #[test]
    fn indexes_iter() {
        let mut builder = IndexesBuilder::new(vec!["foo", "bar"]);
        builder.add(&[IndexValue::from(2_usize), IndexValue::from(3_usize)]);
        builder.add(&[IndexValue::from(1_usize), IndexValue::from(2_usize)]);
        builder.add(&[IndexValue::from(4_usize), IndexValue::from(3_usize)]);

        let idx = builder.finish();
        let mut iter = idx.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[IndexValue::from(2_usize), IndexValue::from(3_usize)]);
        assert_eq!(iter.next().unwrap(), &[IndexValue::from(1_usize), IndexValue::from(2_usize)]);
        assert_eq!(iter.next().unwrap(), &[IndexValue::from(4_usize), IndexValue::from(3_usize)]);
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
        builder.add(&[IndexValue::from(0_usize), IndexValue::from(1_usize)]);
        builder.add(&[IndexValue::from(0_usize), IndexValue::from(1_usize)]);
    }
}
