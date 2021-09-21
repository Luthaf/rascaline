use log::warn;
use ndarray::Array3;

use crate::{Matrix3, Vector3D};
use super::{UnitCell, Pair};

/// `f64::clamp` backported to rust 1.45
fn f64_clamp(mut x: f64, min: f64, max: f64) -> f64 {
    debug_assert!(min <= max);
    if x < min {
        x = min;
    }
    if x > max {
        x = max;
    }
    return x;
}

/// `usize::clamp` backported to rust 1.45
fn usize_clamp(mut x: usize, min: usize, max: usize) -> usize {
    debug_assert!(min <= max);
    if x < min {
        x = min;
    }
    if x > max {
        x = max;
    }
    return x;
}

/// Maximal number of cells, we need to use this to prevent having too many
/// cells with a small unit cell and a large cutoff
const MAX_NUMBER_OF_CELLS: f64 = 1e5;

/// A cell shift represents the displacement along cell axis between the actual
/// position of an atom and a periodic image of this atom.
///
/// The cell shift can be used to reconstruct the vector between two points,
/// wrapped inside the unit cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CellShift([isize; 3]);

impl std::ops::Add<CellShift> for CellShift {
    type Output = CellShift;

    fn add(mut self, rhs: CellShift) -> Self::Output {
        self.0[0] += rhs[0];
        self.0[1] += rhs[1];
        self.0[2] += rhs[2];
        return self;
    }
}

impl std::ops::Sub<CellShift> for CellShift {
    type Output = CellShift;

    fn sub(mut self, rhs: CellShift) -> Self::Output {
        self.0[0] -= rhs[0];
        self.0[1] -= rhs[1];
        self.0[2] -= rhs[2];
        return self;
    }
}

impl std::ops::Index<usize> for CellShift {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl CellShift {
    /// Compute the shift vector in cartesian coordinates, using the given cell
    /// matrix (stored in row major order).
    pub fn cartesian(&self, cell: &Matrix3) -> Vector3D {
        let x = cell[0][0] * self[0] as f64 + cell[1][0] * self[1] as f64 + cell[2][0] * self[2] as f64;
        let y = cell[0][1] * self[0] as f64 + cell[1][1] * self[1] as f64 + cell[2][1] * self[2] as f64;
        let z = cell[0][2] * self[0] as f64 + cell[1][2] * self[1] as f64 + cell[2][2] * self[2] as f64;
        Vector3D::new(x, y, z)
    }
}

/// Pair produced by the cell list. The vector between the atoms can be
/// constructed as `position[second] - position[first] + shift.cartesian(unit_cell)`
#[derive(Debug, Clone)]
pub struct CellPair {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
    /// number of shifts along the cell for this pair
    pub shift: CellShift,
}

/// Data associated with an atoms inside the `CellList`
#[derive(Debug, Clone)]
pub struct AtomData {
    /// index of the atom in the original system
    index: usize,
    /// the shift vector from the actual atom position to the image of this atom
    /// inside the unit cell
    shift: CellShift,
}

/// The cell list is used to sort atoms inside bins/cells.
///
/// The list of potential pairs is then constructed by looking through all
/// neighboring cells (the number of cells to search depends on the cutoff and
/// the size of the cells) for each atom to create pair candidates.
#[derive(Debug, Clone)]
pub struct CellList {
    /// How many cells do we need to look at when searching neighbors to include
    /// all neighbors below cutoff
    n_search: [isize; 3],
    /// the cells themselves
    cells: ndarray::Array3<Vec<AtomData>>,
    /// Unit cell defining periodic boundary conditions
    unit_cell: UnitCell,
}

impl CellList {
    /// Create a new `CellList` for the given unit cell and cutoff, determining
    /// all required parameters.
    pub fn new(unit_cell: UnitCell, cutoff: f64) -> CellList {
        let distances_between_faces = if unit_cell.is_infinite() {
            // use a pseudo orthorhombic cell with size 1, `n_search` below will
            // make sure we look to every cell up to the cutoff
            Vector3D::new(1.0, 1.0, 1.0)
        } else {
            unit_cell.distances_between_faces()
        };

        let mut n_cells = [
            f64_clamp(f64::trunc(distances_between_faces[0] / cutoff), 1.0, f64::INFINITY),
            f64_clamp(f64::trunc(distances_between_faces[1] / cutoff), 1.0, f64::INFINITY),
            f64_clamp(f64::trunc(distances_between_faces[2] / cutoff), 1.0, f64::INFINITY),
        ];

        assert!(n_cells[0].is_finite() && n_cells[1].is_finite() && n_cells[2].is_finite());

        // limit memory consumption by ensuring we have less than `MAX_N_CELLS`
        // cells to look though
        let n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
        if n_cells_total > MAX_NUMBER_OF_CELLS {
            // set the total number of cells close to MAX_N_CELLS, while keeping
            // roughly the ratio of cells in each direction
            let ratio_x_y = n_cells[0] / n_cells[1];
            let ratio_y_z = n_cells[1] / n_cells[2];

            n_cells[2] = f64::trunc(f64::cbrt(MAX_NUMBER_OF_CELLS / (ratio_x_y * ratio_y_z * ratio_y_z)));
            n_cells[1] = f64::trunc(ratio_y_z * n_cells[2]);
            n_cells[0] = f64::trunc(ratio_x_y * n_cells[1]);
        }

        // number of cells to search in each direction to make sure all possible
        // pairs below the cutoff are accounted for.
        let mut n_search = [
            f64::trunc(cutoff * n_cells[0] / distances_between_faces[0]) as isize,
            f64::trunc(cutoff * n_cells[1] / distances_between_faces[1]) as isize,
            f64::trunc(cutoff * n_cells[2] / distances_between_faces[2]) as isize,
        ];

        let n_cells = [
            n_cells[0] as usize,
            n_cells[1] as usize,
            n_cells[2] as usize,
        ];

        for spatial in 0..3 {
            if n_search[spatial] < 1 {
                n_search[spatial] = 1;
            }

            // don't look for neighboring cells if we have only one cell and no
            // periodic boundary condition
            if n_cells[spatial] == 1 && unit_cell.is_infinite() {
                n_search[spatial] = 0;
            }
        }

        CellList {
            n_search: n_search,
            cells: Array3::from_elem(n_cells, Default::default()),
            unit_cell: unit_cell,
        }
    }

    /// Add a single atom to the cell list at the given `position`. The atom is
    /// uniquely identified by its `index`.
    pub fn add_atom(&mut self, index: usize, position: Vector3D) {
        let fractional = if self.unit_cell.is_infinite() {
            position
        } else {
            self.unit_cell.fractional(position)
        };

        let n_cells = self.cells.shape();
        let n_cells = [n_cells[0], n_cells[1], n_cells[2]];

        // find the cell in which this atom should go
        let cell_index = [
            f64::floor(fractional[0] * n_cells[0] as f64) as isize,
            f64::floor(fractional[1] * n_cells[1] as f64) as isize,
            f64::floor(fractional[2] * n_cells[2] as f64) as isize,
        ];

        // deal with pbc by wrapping the atom inside if it was outside of the
        // cell
        let (shift, cell_index) = if self.unit_cell.is_infinite() {
            let cell_index = [
                usize_clamp(cell_index[0] as usize, 0, n_cells[0] - 1),
                usize_clamp(cell_index[1] as usize, 0, n_cells[1] - 1),
                usize_clamp(cell_index[2] as usize, 0, n_cells[2] - 1),
            ];
            ([0, 0, 0], cell_index)
        } else {
            divmod_vec(cell_index, n_cells)
        };

        self.cells[cell_index].push(AtomData {
            index: index,
            shift: CellShift(shift),
        });
    }

    /// Get the list of candidate pair. Some pairs might be separated by more
    /// than `cutoff`, so additional filtering of the pairs might be required
    /// later.
    ///
    /// This function produces a so-called "half" neighbors list, where each
    /// pair is only included once. For example, if atoms 33 and 64 are in range
    /// of each other, the output will only contain pairs in the order 33-64,
    /// and not 64-33.
    ///
    /// If two atoms are neighbors of one another more than once (this can
    /// happen when not using minimal image convention), all pairs at different
    /// distances/directions are still included. Using the example above and
    /// with a cutoff of 5 Å, we can have a pair between atoms 33-64 at 2.6 Å
    /// and another pair between atoms 33-64 at 4.8 Å.
    pub fn pairs(&self) -> Vec<CellPair> {
        let mut pairs = Vec::new();

        let n_cells = self.cells.shape();
        let n_cells = [n_cells[0], n_cells[1], n_cells[2]];

        let search_x = -self.n_search[0]..=self.n_search[0];
        let search_y = -self.n_search[1]..=self.n_search[1];
        let search_z = -self.n_search[2]..=self.n_search[2];

        // for each cell in the cell list
        for ((cell_i_x, cell_i_y, cell_i_z), current_cell) in self.cells.indexed_iter() {
            // look through each neighboring cell
            for delta_x in search_x.clone() {
                for delta_y in search_y.clone() {
                    for delta_z in search_z.clone() {
                        let cell_i = [
                            cell_i_x as isize + delta_x,
                            cell_i_y as isize + delta_y,
                            cell_i_z as isize + delta_z,
                        ];

                        // shift vector from one cell to the other and index of
                        // the neighboring cell
                        let (cell_shift, neighbor_cell_i) = divmod_vec(cell_i, n_cells);

                        for atom_i in current_cell {
                            for atom_j in &self.cells[neighbor_cell_i] {
                                // create a half neighbor list
                                if atom_i.index > atom_j.index {
                                    continue;
                                }

                                let shift = CellShift(cell_shift) + atom_i.shift - atom_j.shift;
                                let shift_is_zero = shift[0] == 0 && shift[1] == 0 && shift[2] == 0;

                                if atom_i.index == atom_j.index && shift_is_zero {
                                    // only create pair with the same atom twice
                                    // if the pair spans more than one unit cell
                                    continue;
                                }

                                if self.unit_cell.is_infinite() && !shift_is_zero {
                                    // do not create pairs crossing the periodic
                                    // boundaries in an infinite cell
                                    continue;
                                }

                                pairs.push(CellPair {
                                    first: atom_i.index,
                                    second: atom_j.index,
                                    shift: shift,
                                });
                            }
                        } // loop over atoms in current neighbor cells

                    }
                }
            } // loop over neighboring cells

        }

        return pairs;
    }
}


/// Function to compute both quotient and remainder of the division of a by b.
/// This function follows Python convention, making sure the remainder have the
/// same sign as `b`.
fn divmod(a: isize, b: usize) -> (isize, usize) {
    let b = b as isize;
    let mut quotient = a / b;
    let mut remainder = a % b;
    if remainder < 0 {
        remainder += b;
        quotient -= 1;
    }
    return (quotient, remainder as usize);
}

/// Apply the [`divmod`] function to three components at the time
fn divmod_vec(a: [isize; 3], b: [usize; 3]) -> ([isize; 3], [usize; 3]) {
    let (qx, rx) = divmod(a[0], b[0]);
    let (qy, ry) = divmod(a[1], b[1]);
    let (qz, rz) = divmod(a[2], b[2]);
    return ([qx, qy, qz], [rx, ry, rz]);
}

/// A neighbor list implementation usable with any system
#[derive(Clone, Debug)]
pub struct NeighborsList {
    /// the cutoff used to create this neighbor list
    pub cutoff: f64,
    /// all pairs in the system
    pub pairs: Vec<Pair>,
    /// all pairs in the system, classified by associated center
    pub pairs_by_center: Vec<Vec<Pair>>,
}

impl NeighborsList {
    #[time_graph::instrument(name = "NeighborsList")]
    pub fn new(positions: &[Vector3D], unit_cell: UnitCell, cutoff: f64) -> NeighborsList {
        let mut cell_list = CellList::new(unit_cell, cutoff);

        for (index, &position) in positions.iter().enumerate() {
            cell_list.add_atom(index, position);
        }

        let cell_matrix = unit_cell.matrix();
        let cutoff2 = cutoff * cutoff;

        // the cell list creates too many pairs, we only need to keep the one where
        // the distance is actually below the cutoff
        let mut pairs = Vec::new();
        let mut pairs_by_center = vec![Vec::new(); positions.len()];

        for pair in cell_list.pairs() {
            let mut vector = positions[pair.second] - positions[pair.first];
            vector += pair.shift.cartesian(&cell_matrix);

            let distance2 = vector * vector;
            if distance2 < cutoff2 {
                if distance2 < 1e-3 {
                    warn!(
                        "atoms {} and {} are very close to one another ({} A)",
                        pair.first, pair.second, distance2.sqrt()
                    );
                }

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

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use crate::Matrix3;

    use super::*;

    #[test]
    fn non_periodic() {
        let positions = [
            Vector3D::new(0.134, 1.282, 1.701),
            Vector3D::new(-0.273, 1.026, -1.471),
            Vector3D::new(1.922, -0.124, 1.900),
            Vector3D::new(1.400, -0.464, 0.480),
            Vector3D::new(0.149, 1.865, 0.635),
        ];

        let neighbors = NeighborsList::new(&positions, UnitCell::infinite(), 3.42);

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

        assert_eq!(neighbors.pairs.len(), reference.len());
        for (pair, reference) in neighbors.pairs.iter().zip(&reference) {
            assert_eq!(pair.first, reference.0);
            assert_eq!(pair.second, reference.1);
            assert_ulps_eq!(pair.distance, reference.2);
        }
    }

    #[test]
    fn fcc_cell() {
        let cell = UnitCell::from(Matrix3::from([
            [0.0, 1.5, 1.5],
            [1.5, 0.0, 1.5],
            [1.5, 1.5, 0.0],
        ]));
        let positions = [Vector3D::new(0.0, 0.0, 0.0)];
        let neighbors = NeighborsList::new(&positions, cell, 3.0);

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

        assert_eq!(neighbors.pairs.len(), 12);
        for (pair, vector) in neighbors.pairs.iter().zip(&expected) {
            assert_eq!(pair.first, 0);
            assert_eq!(pair.second, 0);
            assert_ulps_eq!(pair.distance, 2.1213203435596424);
            assert_ulps_eq!(pair.vector / 1.5, vector);
        }
    }

    #[test]
    fn large_cell_small_cutoff() {
        let cell = UnitCell::cubic(54.0);
        let positions = [
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.0, 2.0, 0.0),
            Vector3D::new(0.0, 0.0, 2.0),
            // atoms outside the cell natural boundaries
            Vector3D::new(-6.0, 0.0, 0.0),
            Vector3D::new(-6.0, -2.0, 0.0),
            Vector3D::new(-6.0, 0.0, -2.0),
        ];

        let neighbors = NeighborsList::new(&positions, cell, 2.1);

        let expected = [
            (0, 1),
            (0, 2),
            (3, 4),
            (3, 5),
        ];

        assert_eq!(neighbors.pairs.len(), expected.len());
        for (pair, expected) in neighbors.pairs.iter().zip(&expected) {
            assert_eq!(pair.first, expected.0);
            assert_eq!(pair.second, expected.1);
            assert_ulps_eq!(pair.distance, 2.0);
        }
    }
}
