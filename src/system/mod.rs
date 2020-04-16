use crate::Vector3D;

mod cell;
pub use self::cell::UnitCell;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub use self::test_utils::test_systems;

pub trait System {
    fn cell(&self) -> UnitCell;
    fn size(&self) -> usize;
    fn species(&self) -> &[usize];
    fn positions(&self) -> &[Vector3D];

    fn compute_neighbors(&mut self, cutoff: f64);

    fn foreach_pair(&self, function: &mut dyn FnMut(usize, usize, f64));
}
