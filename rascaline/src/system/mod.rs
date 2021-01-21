use crate::Vector3D;

mod cell;
pub use self::cell::UnitCell;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub use self::test_utils::test_systems;

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

pub trait System {
    fn cell(&self) -> UnitCell;
    fn size(&self) -> usize;
    fn species(&self) -> &[usize];
    fn positions(&self) -> &[Vector3D];

    fn compute_neighbors(&mut self, cutoff: f64);

    fn pairs(&self) -> &[Pair];
}
