use crate::Vector3D;

mod cell;
pub use self::cell::UnitCell;

mod simple_system;
pub use self::simple_system::SimpleSystem;

mod chemfiles;
pub use self::chemfiles::read_from_file;

#[cfg(test)]
pub(crate) mod test_utils;
#[cfg(test)]
pub(crate) use self::test_utils::test_systems;

/// Pair of atoms coming from a neighbor list.
// WARNING: any change to this definition MUST be reflected in rascal_pair_t as
// well
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Pair {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
    /// vector from the first atom to the second atom, wrapped inside the unit
    /// cell as required
    pub vector: Vector3D,
}

/// A `System` deals with the storage of atoms and related information, as well
/// as the computation of neighbor lists.
pub trait System {
    /// Get the unit cell for this system
    fn cell(&self) -> UnitCell;

    /// Get the number of atoms in this system
    fn size(&self) -> usize;

    /// Get the atomic species for all atoms in this system. The returned value
    /// must be a slice of length `self.size()`, where each different atomic
    /// species is identified with a different usize value. These values are
    /// usually the atomic number, but don't have to.
    fn species(&self) -> &[usize];

    /// Get the positions for all atoms in this system. The returned value must
    /// be a slice of length `self.size()` containing the cartesian coordinates
    /// of all atoms in the system.
    fn positions(&self) -> &[Vector3D];

    /// Compute the neighbor list according to the given cutoff, and store it
    /// for later access with `pairs` or `pairs_around`.
    fn compute_neighbors(&mut self, cutoff: f64);

    /// Get the list of pairs in this system. This list of pair should only
    /// contain each pair once (and not twice as `i-j` and `j-i`), should not
    /// contain self pairs (`i-i`); and should only contains pairs where the
    /// distance between atoms is actually bellow the cutoff passed in the last
    /// call to `compute_neighbors`. This function is only valid to call after a
    /// call to `compute_neighbors`.
    fn pairs(&self) -> &[Pair];

    /// Get the list of pairs in this system which include the atom at index
    /// `center`. The same restrictions on the list of pairs as `System::pairs`
    /// applies, with the additional condition that the pair `i-j` should be
    /// included both in the return of `pairs_containing(i)` and
    /// `pairs_containing(j)`.
    fn pairs_containing(&self, center: usize) -> &[Pair];
}
