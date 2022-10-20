use std::collections::BTreeSet;
use std::sync::Arc;

use equistore::Labels;

use crate::{Error, System};

/// Atomic species filters to be used when building samples and gradient sample
pub enum SpeciesFilter {
    /// Any atomic species is fine
    Any,
    /// Only the given atomic species should match
    Single(i32),
    /// Any of the given atomic species is fine
    OneOf(Vec<i32>),
    /// All of the given atoms species must be present. This can only be used
    /// for neighbor species selection.
    AllOf(BTreeSet<i32>),
}

impl SpeciesFilter {
    /// Check if a given species matches the filter
    pub fn matches(&self, species: i32) -> bool {
        match self {
            SpeciesFilter::Any => true,
            SpeciesFilter::Single(selected) => species == *selected,
            SpeciesFilter::OneOf(selected) => selected.contains(&species),
            SpeciesFilter::AllOf(_) => panic!("internal error: can not call `matches` on a `SpeciesFilter::AllOf`"),
        }
    }
}

/// Abstraction over the different kinds of samples used in rascaline.
///
/// Different implementations of this trait correspond to different types of
/// samples (for example one sample for each structure; or one sample for each
/// pair, etc.)
///
/// Each implementation must be able to generate samples from a list of systems,
/// and can optionally implement samples for gradients calculation.
pub trait SamplesBuilder {
    /// Get the names used by the samples
    fn samples_names() -> Vec<&'static str>;

    /// Create `Labels` containing all the samples corresponding to the given
    /// list of systems.
    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Arc<Labels>, Error>;

    /// Create a set of `Labels` containing the gradient samples corresponding
    /// to the given `samples` in the given `systems`; and only these.
    #[allow(unused_variables)]
    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Labels) -> Result<Arc<Labels>, Error>;
}


mod atom_centered;
pub use self::atom_centered::AtomCenteredSamples;

mod long_range;
pub use self::long_range::LongRangePerAtom;
