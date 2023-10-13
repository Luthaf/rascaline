use std::collections::BTreeMap;

use crate::{Error, Vector3D};

mod cell;
pub use self::cell::{UnitCell, CellShape};

mod neighbors;
pub use self::neighbors::NeighborsList;

mod simple_system;
pub use self::simple_system::SimpleSystem;

mod chemfiles;
pub use self::chemfiles::read_from_file;
pub use self::chemfiles::read_simple_systems_from_file;

#[cfg(test)]
pub(crate) mod test_utils;

/// Pair of atoms coming from a neighbor list.
// WARNING: any change to this definition MUST be reflected in rascal_pair_t as
// well
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
    /// distance between the two atoms
    pub distance: f64,
    /// vector from the first atom to the second atom, accounting for periodic
    /// boundary conditions. This should be `position\[second\] -
    /// position\[first\] + H * cell_shift` where `H` is the cell matrix.
    pub vector: Vector3D,
    /// How many cell shift where applied to the `second` atom to create this
    /// pair.
    pub cell_shift_indices: [i32; 3],
}

/// A `SystemBase` deals with the storage of atoms and related information, as well
/// as the computation of neighbor lists.
pub trait SystemBase: Send + Sync {
    /// Get the unit cell for this system
    fn cell(&self) -> Result<UnitCell, Error>;

    /// Get the number of atoms in this system
    fn size(&self) -> Result<usize, Error>;

    /// Get the atomic types for all atoms in this system. The returned value
    /// must be a slice of length `self.size()`, where each different atomic
    /// type is identified with a different integer value. These values are
    /// usually the atomic number, but don't have to.
    fn types(&self) -> Result<&[i32], Error>;

    /// Get the positions for all atoms in this system. The returned value must
    /// be a slice of length `self.size()` containing the Cartesian coordinates
    /// of all atoms in the system.
    fn positions(&self) -> Result<&[Vector3D], Error>;

    /// Compute the neighbor list according to the given cutoff, and store it
    /// for later access with `pairs` or `pairs_around`.
    fn compute_neighbors(&mut self, cutoff: f64) -> Result<(), Error>;

    /// Get the list of pairs in this system. This list of pair should only
    /// contain each pair once (and not twice as `i-j` and `j-i`), should not
    /// contain self pairs (`i-i`); and should only contains pairs where the
    /// distance between atoms is actually bellow the cutoff passed in the last
    /// call to `compute_neighbors`. This function is only valid to call after a
    /// call to `compute_neighbors`.
    fn pairs(&self) -> Result<&[Pair], Error>;

    /// Get the list of pairs in this system which include the atom at the given
    /// index. The same restrictions on the list of pairs as `System::pairs`
    /// applies, with the additional condition that the pair `i-j` should be
    /// included both in the return of `pairs_containing(i)` and
    /// `pairs_containing(j)`.
    fn pairs_containing(&self, atom: usize) -> Result<&[Pair], Error>;
}

/// TODO
pub struct System {
    implementation: Box<dyn SystemBase>,
    data: BTreeMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl std::ops::Deref for System {
    type Target = dyn SystemBase + 'static;

    fn deref(&self) -> &Self::Target {
        &*self.implementation
    }
}

impl std::ops::DerefMut for System {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.implementation
    }
}

impl System {
    /// TODO
    pub fn new(system: impl SystemBase + 'static) -> System {
        System {
            implementation: Box::new(system),
            data: BTreeMap::new()
        }
    }

    /// TODO
    pub fn store_data(&mut self, name: String, data: impl std::any::Any + Send + Sync + 'static) {
        self.data.insert(name, Box::new(data));
    }

    /// TODO
    pub fn data(&self, name: &str) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.data.get(name).map(|v| &**v)
    }

    /// TODO
    pub fn data_mut(&mut self, name: &str) -> Option<&mut (dyn std::any::Any + Send + Sync)> {
        self.data.get_mut(name).map(|v| &mut **v)
    }
}
