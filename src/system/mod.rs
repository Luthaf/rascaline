use crate::Vector3D;

mod cell;
pub use self::cell::UnitCell;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub use self::test_utils::test_systems;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub first: usize,
    pub second: usize,
    pub distance: f64,
}

pub trait System {
    fn cell(&self) -> UnitCell;
    fn size(&self) -> usize;
    fn species(&self) -> &[usize];
    fn positions(&self) -> &[Vector3D];

    fn compute_neighbors(&mut self, cutoff: f64);

    fn pairs(&self) -> &[Pair];
}
