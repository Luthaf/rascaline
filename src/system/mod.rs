use crate::Vector3D;

mod crappy_neighbors_list;
use self::crappy_neighbors_list::CrappyNeighborsList;

mod cell;
use self::cell::UnitCell;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub use self::test_utils::test_system;

pub trait NeighborsList {
    fn natoms(&self) -> usize;
    fn foreach_pairs(&self, function: &mut dyn FnMut(usize, usize));
}

pub trait System {
    fn cell(&self) -> UnitCell;
    fn natoms(&self) -> usize;
    fn types(&self) -> &[usize];
    fn positions(&self) -> &[Vector3D];
    fn neighbors(&self, cutoff: f64) -> Box<dyn NeighborsList> {
        Box::new(CrappyNeighborsList::new(self, cutoff))
    }
}
