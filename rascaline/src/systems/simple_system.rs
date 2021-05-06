use super::{UnitCell, System, Vector3D, Pair};

use super::neighbors::CellList;

#[derive(Clone, Debug)]
struct NeighborsList {
    cutoff: f64,
    pairs: Vec<Pair>,
    pairs_by_center: Vec<Vec<Pair>>,
}

impl NeighborsList {
    #[time_graph::instrument(name = "neighbor list")]
    pub fn new<S: System + ?Sized>(system: &S, cutoff: f64) -> NeighborsList {
        let unit_cell = system.cell();
        let mut cell_list = CellList::new(unit_cell, cutoff);

        let positions = system.positions();
        for (index, &position) in positions.iter().enumerate() {
            cell_list.add_atom(index, position);
        }

        let cell_matrix = unit_cell.matrix();
        let cutoff2 = cutoff * cutoff;

        // the cell list creates too many pairs, we only need to keep the one where
        // the distance is actually below the cutoff
        let mut pairs = Vec::new();
        let mut pairs_by_center = vec![Vec::new(); system.size()];

        for pair in cell_list.pairs() {
            let mut vector = positions[pair.second] - positions[pair.first];
            vector += pair.shift.dot(&cell_matrix);

            let distance2 = vector * vector;
            if distance2 < cutoff2 {
                let pair = Pair {
                    first: pair.first,
                    second: pair.second,
                    distance: distance2.sqrt(),
                    vector: vector,
                };

                pairs.push(pair);
                pairs_by_center[pair.first].push(pair);
                pairs_by_center[pair.second].push(pair);
            }
        }

        // sort the pairs to make sure the final output of rascaline is ordered
        // naturally
        pairs.sort_unstable_by_key(|pair| (pair.first, pair.second));
        for pairs in &mut pairs_by_center {
            pairs.sort_unstable_by_key(|pair| (pair.first, pair.second));
        }

        return NeighborsList {
            cutoff: cutoff,
            pairs: pairs,
            pairs_by_center: pairs_by_center,
        };
    }
}

/// A simple implementation of `System` to use when no other is available
#[derive(Clone, Debug)]
pub struct SimpleSystem {
    cell: UnitCell,
    species: Vec<usize>,
    positions: Vec<Vector3D>,
    neighbors: Option<NeighborsList>,
}

impl SimpleSystem {
    /// Create a new empty system with the given unit cell
    pub fn new(cell: UnitCell) -> SimpleSystem {
        SimpleSystem {
            cell: cell,
            species: Vec::new(),
            positions: Vec::new(),
            neighbors: None,
        }
    }

    /// Add an atom with the given species and position to this system
    pub fn add_atom(&mut self, species: usize, position: Vector3D) {
        self.species.push(species);
        self.positions.push(position);
    }

    #[cfg(test)]
    pub(crate) fn positions_mut(&mut self) -> &mut [Vector3D] {
        // any position access invalidates the neighbor list
        self.neighbors = None;
        return &mut self.positions;
    }
}

impl System for SimpleSystem {
    fn size(&self) -> usize {
        self.species.len()
    }

    fn positions(&self) -> &[Vector3D] {
        &self.positions
    }

    fn species(&self) -> &[usize] {
        &self.species
    }

    fn cell(&self) -> UnitCell {
        self.cell
    }

    #[allow(clippy::float_cmp)]
    fn compute_neighbors(&mut self, cutoff: f64) {
        // re-use already computed NL is possible
        if let Some(ref nl) = self.neighbors {
            if nl.cutoff == cutoff {
                return;
            }
        }

        self.neighbors = Some(NeighborsList::new(self, cutoff));
    }

    fn pairs(&self) -> &[Pair] {
        &self.neighbors.as_ref().expect("neighbor list is not initialized").pairs
    }

    fn pairs_containing(&self, center: usize) -> &[Pair] {
        &self.neighbors.as_ref().expect("neighbor list is not initialized").pairs_by_center[center]
    }
}

impl From<&dyn System> for SimpleSystem {
    fn from(system: &dyn System) -> SimpleSystem {
        let mut new = SimpleSystem::new(system.cell());
        for (&species, &position) in system.species().iter().zip(system.positions()) {
            new.add_atom(species, position);
        }
        return new;
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

        assert_eq!(system.size(), 3);
        assert_eq!(system.species.len(), 3);
        assert_eq!(system.positions.len(), 3);

        assert_eq!(system.species(), &[3, 1, 3]);
        assert_eq!(system.positions(), &[
            Vector3D::new(2.0, 3.0, 4.0),
            Vector3D::new(1.0, 3.0, 4.0),
            Vector3D::new(5.0, 3.0, 4.0),
        ]);
    }

    #[allow(clippy::wildcard_imports)]
    mod neighbors {
        use approx::assert_ulps_eq;

        use crate::Matrix3;

        use super::super::*;

        #[test]
        fn non_periodic() {
            let mut system = SimpleSystem::new(UnitCell::infinite());
            system.add_atom(1, Vector3D::new(0.134, 1.282, 1.701));
            system.add_atom(1, Vector3D::new(-0.273, 1.026, -1.471));
            system.add_atom(1, Vector3D::new(1.922, -0.124, 1.900));
            system.add_atom(1, Vector3D::new(1.400, -0.464, 0.480));
            system.add_atom(1, Vector3D::new(0.149, 1.865, 0.635));

            system.compute_neighbors(3.42);
            let pairs = system.pairs();

            // reference computed with ASE
            let reference = [
                (0, 1, 3.2082345612501593),
                (0, 2, 2.283282943482914),
                (0, 3, 2.4783286706972505),
                (0, 4, 1.215100818862369),
                (1, 3, 2.9707625283755013),
                (1, 4, 2.3059143522689647),
                (2, 3, 1.550639867925496),
                (2, 4, 2.9495550511899244),
                (3, 4, 2.6482573515427084),
            ];

            assert_eq!(pairs.len(), reference.len());
            for (pair, reference) in pairs.iter().zip(&reference) {
                assert_eq!(pair.first, reference.0);
                assert_eq!(pair.second, reference.1);
                assert_ulps_eq!(pair.distance, reference.2);
            }
        }

        #[test]
        fn fcc_cell() {
            let mut system = SimpleSystem::new(UnitCell::from(Matrix3::from([
                [0.0, 1.5, 1.5],
                [1.5, 0.0, 1.5],
                [1.5, 1.5, 0.0],
            ])));
            system.add_atom(1, Vector3D::new(0.0, 0.0, 0.0));

            system.compute_neighbors(3.0);
            let pairs = system.pairs();

            let expected = [
                Vector3D::new(0.0, -1.0, -1.0),
                Vector3D::new(1.0, 0.0, -1.0),
                Vector3D::new(1.0, -1.0, 0.0),
                Vector3D::new(-1.0, 0.0, -1.0),
                Vector3D::new(0.0, 1.0, -1.0),
                Vector3D::new(-1.0, -1.0, 0.0),
                Vector3D::new(1.0, 1.0, 0.0),
                Vector3D::new(0.0, -1.0, 1.0),
                Vector3D::new(1.0, 0.0, 1.0),
                Vector3D::new(-1.0, 1.0, 0.0),
                Vector3D::new(-1.0, 0.0, 1.0),
                Vector3D::new(0.0, 1.0, 1.0),
            ];

            assert_eq!(pairs.len(), 12);
            for (pair, vector) in pairs.iter().zip(&expected) {
                assert_eq!(pair.first, 0);
                assert_eq!(pair.second, 0);
                assert_ulps_eq!(pair.distance, 2.1213203435596424);
                assert_ulps_eq!(pair.vector / 1.5, vector);
            }
        }

        #[test]
        fn large_cell_small_cutoff() {
            let mut system = SimpleSystem::new(UnitCell::cubic(54.0));
            system.add_atom(1, Vector3D::new(0.0, 0.0, 0.0));
            system.add_atom(1, Vector3D::new(0.0, 2.0, 0.0));
            system.add_atom(1, Vector3D::new(0.0, 0.0, 2.0));

            // atoms outside the cell natural boundaries
            system.add_atom(1, Vector3D::new(-6.0, 0.0, 0.0));
            system.add_atom(1, Vector3D::new(-6.0, -2.0, 0.0));
            system.add_atom(1, Vector3D::new(-6.0, 0.0, -2.0));

            system.compute_neighbors(2.1);
            let pairs = system.pairs();

            let expected = [
                (0, 1),
                (0, 2),
                (3, 4),
                (3, 5),
            ];

            assert_eq!(pairs.len(), expected.len());
            for (pair, expected) in pairs.iter().zip(&expected) {
                assert_eq!(pair.first, expected.0);
                assert_eq!(pair.second, expected.1);
                assert_ulps_eq!(pair.distance, 2.0);
            }
        }
    }
}
