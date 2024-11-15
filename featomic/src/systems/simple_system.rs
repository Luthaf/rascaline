use crate::Error;

use super::{UnitCell, System, Vector3D, Pair};

use super::neighbors::NeighborsList;

/// A simple implementation of `System` to use when no other is available
#[derive(Clone, Debug)]
pub struct SimpleSystem {
    pub(crate) cell: UnitCell,
    types: Vec<i32>,
    positions: Vec<Vector3D>,
    neighbors: Option<NeighborsList>,
}

impl SimpleSystem {
    /// Create a new empty system with the given unit cell
    pub fn new(cell: UnitCell) -> SimpleSystem {
        SimpleSystem {
            cell: cell,
            types: Vec::new(),
            positions: Vec::new(),
            neighbors: None,
        }
    }

    /// Add an atom with the given atomic type and position to this system
    pub fn add_atom(&mut self, atomic_type: i32, position: Vector3D) {
        self.types.push(atomic_type);
        self.positions.push(position);
    }

    #[cfg(test)]
    pub(crate) fn positions_mut(&mut self) -> &mut [Vector3D] {
        // any position access invalidates the neighbor list
        self.neighbors = None;
        return &mut self.positions;
    }

    #[cfg(test)]
    pub(crate) fn set_cell(&mut self, cell: UnitCell) {
        // cell change invalidate the neighbor list
        self.neighbors = None;
        self.cell = cell;
    }
}

impl System for SimpleSystem {
    fn size(&self) -> Result<usize, Error> {
        Ok(self.types.len())
    }

    fn positions(&self) -> Result<&[Vector3D], Error> {
        Ok(&self.positions)
    }

    fn types(&self) -> Result<&[i32], Error> {
        Ok(&self.types)
    }

    fn cell(&self) -> Result<UnitCell, Error> {
        Ok(self.cell)
    }

    #[allow(clippy::float_cmp)]
    fn compute_neighbors(&mut self, cutoff: f64) -> Result<(), Error> {
        // re-use already computed NL is possible
        if let Some(ref nl) = self.neighbors {
            if nl.cutoff == cutoff {
                return Ok(());
            }
        }

        self.neighbors = Some(NeighborsList::new(self.positions()?, self.cell()?, cutoff));
        Ok(())
    }

    fn pairs(&self) -> Result<&[Pair], Error> {
        let neighbors = self.neighbors.as_ref().ok_or_else(|| Error::Internal(
            "neighbor list is not initialized".into()
        ))?;
        Ok(&neighbors.pairs)
    }

    fn pairs_containing(&self, atom: usize) -> Result<&[Pair], Error> {
        let neighbors = self.neighbors.as_ref().ok_or_else(|| Error::Internal(
            "neighbor list is not initialized".into()
        ))?;
        Ok(&neighbors.pairs_by_atom[atom])
    }
}

impl std::convert::TryFrom<&dyn System> for SimpleSystem {
    type Error = Error;

    fn try_from(system: &dyn System) -> Result<SimpleSystem, Error> {
        let mut new = SimpleSystem::new(system.cell()?);
        for (&atomic_type, &position) in system.types()?.iter().zip(system.positions()?) {
            new.add_atom(atomic_type, position);
        }
        return Ok(new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_atoms() {
        let mut system = SimpleSystem::new(UnitCell::cubic(10.0));
        system.add_atom(3, Vector3D::new(2.0, 3.0, 4.0));
        system.add_atom(1, Vector3D::new(1.0, 3.0, 4.0));
        system.add_atom(3, Vector3D::new(5.0, 3.0, 4.0));

        assert_eq!(system.size().unwrap(), 3);
        assert_eq!(system.types.len(), 3);
        assert_eq!(system.positions.len(), 3);

        assert_eq!(system.types().unwrap(), &[3, 1, 3]);
        assert_eq!(system.positions().unwrap(), &[
            Vector3D::new(2.0, 3.0, 4.0),
            Vector3D::new(1.0, 3.0, 4.0),
            Vector3D::new(5.0, 3.0, 4.0),
        ]);
    }
}
