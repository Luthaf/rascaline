use std::collections::BTreeSet;

use metatensor::Labels;

use crate::{Error, System};

/// Atomic type filters to be used when building samples and gradient sample
pub enum AtomicTypeFilter {
    /// Any atomic type is fine
    Any,
    /// Only the given atomic type should match
    Single(i32),
    /// Any of the given atomic type is fine
    OneOf(Vec<i32>),
    /// All of the given atoms types must be present. This can only be used
    /// for neighbor types selection.
    AllOf(BTreeSet<i32>),
}

impl AtomicTypeFilter {
    /// Check if a given type matches the filter
    pub fn matches(&self, atomic_type: i32) -> bool {
        match self {
            AtomicTypeFilter::Any => true,
            AtomicTypeFilter::Single(selected) => atomic_type == *selected,
            AtomicTypeFilter::OneOf(selected) => selected.contains(&atomic_type),
            AtomicTypeFilter::AllOf(_) => panic!("internal error: can not call `matches` on a `AtomicTypeFilter::AllOf`"),
        }
    }
}

/// Abstraction over the different kinds of samples used in rascaline.
///
/// Different implementations of this trait correspond to different types of
/// samples (for example one sample for each system; or one sample for each
/// pair, etc.)
///
/// Each implementation must be able to generate samples from a list of systems,
/// and can optionally implement samples for gradients calculation.
pub trait SamplesBuilder {
    /// Get the names used by the samples
    fn sample_names() -> Vec<&'static str>;

    /// Create `Labels` containing all the samples corresponding to the given
    /// list of systems.
    fn samples(&self, systems: &mut [System]) -> Result<Labels, Error>;

    /// Create a set of `Labels` containing the gradient samples corresponding
    /// to the given `samples` in the given `systems`; and only these.
    #[allow(unused_variables)]
    fn gradients_for(&self, systems: &mut [System], samples: &Labels) -> Result<Labels, Error>;
}


mod atom_centered;
pub use self::atom_centered::AtomCenteredSamples;

mod long_range;
pub use self::long_range::LongRangeSamplesPerAtom;
