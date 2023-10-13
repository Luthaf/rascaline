use std::collections::BTreeSet;

use metatensor::TensorMap;
use metatensor::{Labels, LabelsBuilder, LabelValue};

use super::CalculatorBase;

use crate::{Error, System};


/// This calculator computes the neighbor list for a given spherical cutoff, and
/// returns the list of distance vectors between all pairs of atoms strictly
/// inside the cutoff.
///
/// Users can request either a "full" neighbor list (including an entry for both
/// `i - j` pairs and `j - i` pairs) or save memory/computational by only
/// working with "half" neighbor list (only including one entry for each `i/j`
/// pair)
///
/// Pairs between an atom and it's own periodic copy can appear when the cutoff
/// is larger than the cell under periodic boundary conditions. Self pairs with
/// a distance of 0 (i.e. self pairs inside the original unit cell) are only
/// included when using `self_pairs = true`.
///
/// This calculator produces a single property (`"distance"`) with three
/// components (`"pair_xyz"`) containing the x, y, and z component of the
/// distance vector of the pair.
///
/// The samples contain the two atoms indexes, as well as the number of cell
/// boundaries crossed to create this pair.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct NeighborList {
    /// Spherical cutoff to use to determine if two atoms are neighbors
    pub cutoff: f64,
    /// Should we compute a full neighbor list (each pair appears twice, once as
    /// `i-j` and once as `j-i`), or a half neighbor list (each pair only
    /// appears once)
    pub full_neighbor_list: bool,
    /// Should individual atoms be considered their own neighbor? Setting this
    /// to `true` will add "self pairs", i.e. pairs between an atom and itself,
    /// with the distance 0.
    pub self_pairs: bool,
}

/// Sort a pair and return true if the pair was inverted
fn sort_pair((i, j): (i32, i32)) -> ((i32, i32), bool) {
    if i <= j {
        ((i, j), false)
    } else {
        ((j, i), true)
    }
}

impl CalculatorBase for NeighborList {
    fn name(&self) -> String {
        "neighbors list".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        std::slice::from_ref(&self.cutoff)
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        if self.full_neighbor_list {
            FullNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.keys(systems)
        } else {
            HalfNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.keys(systems)
        }
    }

    fn sample_names(&self) -> Vec<&str> {
        return vec!["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        if self.full_neighbor_list {
            FullNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.samples(keys, systems)
        } else {
            HalfNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.samples(keys, systems)
        }
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            // TODO: add support for cell gradients
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, _keys: &Labels, samples: &[Labels], _systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for block_samples in samples {
            let mut builder = LabelsBuilder::new(vec!["sample", "system", "atom"]);
            for (sample_i, &[system_i, first, second, cell_a, cell_b, cell_c]) in block_samples.iter_fixed_size().enumerate() {
                // self pairs do not contribute to gradients
                if first == second && cell_a == 0 && cell_b == 0 && cell_c == 0 {
                    continue;
                }
                builder.add(&[sample_i.into(), system_i, first]);
                builder.add(&[sample_i.into(), system_i, second]);
            }

            results.push(builder.finish());
        }

        return Ok(results);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        let components = vec![Labels::new(["pair_xyz"], &[[0], [1], [2]])];
        return vec![components; keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["distance"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        properties.add(&[LabelValue::new(1)]);
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "NeighborList::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        if self.full_neighbor_list {
            FullNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.compute(systems, descriptor)
        } else {
            HalfNeighborList {
                cutoff: self.cutoff,
                self_pairs: self.self_pairs,
            }.compute(systems, descriptor)
        }
    }
}

/// Implementation of half neighbor list, only including pairs once (such that
/// `types[atom_i] <= types[atom_j]`)
#[derive(Debug, Clone)]
struct HalfNeighborList {
    cutoff: f64,
    self_pairs: bool,
}

impl HalfNeighborList {
    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        let mut all_types_pairs = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;

            let types = system.types()?;
            for pair in system.pairs()? {
                let (types_pair, _) = sort_pair((types[pair.first], types[pair.second]));
                all_types_pairs.insert(types_pair);
            }

            // make sure we have self-pairs keys even if the system does not
            // contain any neighbors with the same atomic type
            if self.self_pairs {
                for &atomic_type in types {
                    all_types_pairs.insert((atomic_type, atomic_type));
                }
            }
        }

        let mut keys = LabelsBuilder::new(vec!["first_atom_type", "second_atom_type"]);
        for (first, second) in all_types_pairs {
            keys.add(&[first, second]);
        }

        return Ok(keys.finish());
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for [first_atom_type, second_atom_type] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(vec![
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c"
            ]);

            for (system_i, system) in systems.iter_mut().enumerate() {
                system.compute_neighbors(self.cutoff)?;
                let types = system.types()?;

                for pair in system.pairs()? {
                    let ((type_i, type_j), invert) = sort_pair((types[pair.first], types[pair.second]));

                    let shifts = pair.cell_shift_indices;
                    let (cell_a, cell_b, cell_c) = if invert {
                        (-shifts[0], -shifts[1], -shifts[2])
                    } else {
                        (shifts[0], shifts[1], shifts[2])
                    };

                    let (atom_i, atom_j) = if invert {
                        (pair.second, pair.first)
                    } else {
                        (pair.first, pair.second)
                    };

                    if type_i == first_atom_type.i32() && type_j == second_atom_type.i32() {
                        builder.add(&[
                            LabelValue::from(system_i),
                            LabelValue::from(atom_i),
                            LabelValue::from(atom_j),
                            LabelValue::from(cell_a),
                            LabelValue::from(cell_b),
                            LabelValue::from(cell_c),
                        ]);
                    }
                }

                // handle self pairs
                if self.self_pairs && first_atom_type == second_atom_type {
                    for center_i in 0..system.size()? {
                        if types[center_i] == first_atom_type.i32() {
                            builder.add(&[
                                system_i.into(),
                                center_i.into(),
                                center_i.into(),
                                LabelValue::from(0),
                                LabelValue::from(0),
                                LabelValue::from(0),
                            ]);
                        }
                    }
                }
            }

            results.push(builder.finish());
        }

        return Ok(results);
    }

    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let types = system.types()?;

            for pair in system.pairs()? {
                // Sort the atomic types in the pair to ensure a canonical order of
                // the atoms in it. This guarantee that multiple call to this
                // calculator always returns pairs in the same order, even if
                // the underlying neighbor list implementation (which comes from
                // the systems) changes.
                //
                // The `invert` variable tells us if we need to invert the pair
                // vector or not.
                let ((type_i, type_j), invert) = sort_pair((types[pair.first], types[pair.second]));

                let pair_vector = if invert {
                    -pair.vector
                } else {
                    pair.vector
                };

                let shifts = pair.cell_shift_indices;
                let (cell_a, cell_b, cell_c) = if invert {
                    (-shifts[0], -shifts[1], -shifts[2])
                } else {
                    (shifts[0], shifts[1], shifts[2])
                };

                let (atom_i, atom_j) = if invert {
                    (pair.second, pair.first)
                } else {
                    (pair.first, pair.second)
                };

                let block_i = descriptor.keys().position(&[
                    type_i.into(), type_j.into()
                ]);

                if let Some(block_i) = block_i {
                    let mut block = descriptor.block_mut_by_id(block_i);
                    let block_data = block.data_mut();

                    let sample_i = block_data.samples.position(&[
                        LabelValue::from(system_i),
                        LabelValue::from(atom_i),
                        LabelValue::from(atom_j),
                        LabelValue::from(cell_a),
                        LabelValue::from(cell_b),
                        LabelValue::from(cell_c),
                    ]);

                    if let Some(sample_i) = sample_i {
                        let array = block_data.values.to_array_mut();
                        for (property_i, &[distance]) in block_data.properties.iter_fixed_size().enumerate() {
                            if distance == 1 {
                                array[[sample_i, 0, property_i]] = pair_vector[0];
                                array[[sample_i, 1, property_i]] = pair_vector[1];
                                array[[sample_i, 2, property_i]] = pair_vector[2];
                            }
                        }

                        if let Some(mut gradient) = block.gradient_mut("positions") {
                            let gradient = gradient.data_mut();

                            let first_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), atom_i.into()
                            ]).expect("missing gradient sample");
                            let second_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), atom_j.into()
                            ]).expect("missing gradient sample");

                            let array = gradient.values.to_array_mut();

                            for (property_i, &[distance]) in gradient.properties.iter_fixed_size().enumerate() {
                                if distance == 1 {
                                    array[[first_grad_sample_i, 0, 0, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 1, 1, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 2, 2, property_i]] = -1.0;

                                    array[[second_grad_sample_i, 0, 0, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 1, 1, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 2, 2, property_i]] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}

/// Implementation of full neighbor list, including each pair twice (once as i-j
/// and once as j-i).
#[derive(Debug, Clone)]
pub struct FullNeighborList {
    pub cutoff: f64,
    pub self_pairs: bool,
}

impl FullNeighborList {
    /// Get the list of keys for these systems (list of pair types present in the systems)
    pub(crate) fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        let mut all_types_pairs = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;

            let types = system.types()?;
            for pair in system.pairs()? {
                all_types_pairs.insert((types[pair.first], types[pair.second]));
                all_types_pairs.insert((types[pair.second], types[pair.first]));
            }

            // make sure we have self-pairs keys even if the system does not
            // contain any neighbors with the same atomic type
            if self.self_pairs {
                for &atomic_type in types {
                    all_types_pairs.insert((atomic_type, atomic_type));
                }
            }
        }

        let mut keys = LabelsBuilder::new(vec!["first_atom_type", "second_atom_type"]);
        for (first, second) in all_types_pairs {
            keys.add(&[first, second]);
        }

        return Ok(keys.finish());
    }

    pub(crate) fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for &[first_atom_type, second_atom_type] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(vec![
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c"
            ]);

            for (system_i, system) in systems.iter_mut().enumerate() {
                system.compute_neighbors(self.cutoff)?;
                let types = system.types()?;

                for pair in system.pairs()? {
                    let cell_a = pair.cell_shift_indices[0];
                    let cell_b = pair.cell_shift_indices[1];
                    let cell_c = pair.cell_shift_indices[2];

                    if first_atom_type == second_atom_type {
                        // same type for both atoms in the pair, add the pair
                        // twice in both directions.
                        if types[pair.first] == first_atom_type.i32() && types[pair.second] == second_atom_type.i32() {
                            builder.add(&[
                                LabelValue::from(system_i),
                                LabelValue::from(pair.first),
                                LabelValue::from(pair.second),
                                LabelValue::from(cell_a),
                                LabelValue::from(cell_b),
                                LabelValue::from(cell_c),
                            ]);

                            builder.add(&[
                                LabelValue::from(system_i),
                                LabelValue::from(pair.second),
                                LabelValue::from(pair.first),
                                LabelValue::from(-cell_a),
                                LabelValue::from(-cell_b),
                                LabelValue::from(-cell_c),
                            ]);
                        }
                    } else {
                        // different types, find the right order for the pair
                        if types[pair.first] == first_atom_type.i32() && types[pair.second] == second_atom_type.i32() {
                            builder.add(&[
                                LabelValue::from(system_i),
                                LabelValue::from(pair.first),
                                LabelValue::from(pair.second),
                                LabelValue::from(cell_a),
                                LabelValue::from(cell_b),
                                LabelValue::from(cell_c),
                            ]);
                        } else if types[pair.second] == first_atom_type.i32() && types[pair.first] == second_atom_type.i32() {
                            builder.add(&[
                                LabelValue::from(system_i),
                                LabelValue::from(pair.second),
                                LabelValue::from(pair.first),
                                LabelValue::from(-cell_a),
                                LabelValue::from(-cell_b),
                                LabelValue::from(-cell_c),
                            ]);
                        }
                    }
                }

                // handle self pairs
                if self.self_pairs && first_atom_type == second_atom_type {
                    for center_i in 0..system.size()? {
                        if types[center_i] == first_atom_type.i32() {
                            builder.add(&[
                                system_i.into(),
                                center_i.into(),
                                center_i.into(),
                                LabelValue::from(0),
                                LabelValue::from(0),
                                LabelValue::from(0),
                            ]);
                        }
                    }
                }
            }

            results.push(builder.finish());
        }

        return Ok(results);
    }

    #[allow(clippy::too_many_lines)]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let types = system.types()?;

            for pair in system.pairs()? {
                if pair.first == pair.second {
                    // self pairs should not be part of the neighbors list
                    assert_ne!(pair.cell_shift_indices, [0, 0, 0]);
                }

                let first_block_i = descriptor.keys().position(&[
                    types[pair.first].into(), types[pair.second].into()
                ]);

                let second_block_i = descriptor.keys().position(&[
                    types[pair.second].into(), types[pair.first].into()
                ]);

                let cell_a = pair.cell_shift_indices[0];
                let cell_b = pair.cell_shift_indices[1];
                let cell_c = pair.cell_shift_indices[2];

                // first, the pair first -> second
                if let Some(first_block_i) = first_block_i {
                    let mut block = descriptor.block_mut_by_id(first_block_i);
                    let block_data = block.data_mut();

                    let sample_i = block_data.samples.position(&[
                        LabelValue::from(system_i),
                        LabelValue::from(pair.first),
                        LabelValue::from(pair.second),
                        LabelValue::from(cell_a),
                        LabelValue::from(cell_b),
                        LabelValue::from(cell_c),
                    ]);

                    if let Some(sample_i) = sample_i {
                        let array = block_data.values.to_array_mut();

                        for (property_i, &[distance]) in block_data.properties.iter_fixed_size().enumerate() {
                            if distance == 1 {
                                array[[sample_i, 0, property_i]] = pair.vector[0];
                                array[[sample_i, 1, property_i]] = pair.vector[1];
                                array[[sample_i, 2, property_i]] = pair.vector[2];
                            }
                        }

                        if let Some(mut gradient) = block.gradient_mut("positions") {
                            let gradient = gradient.data_mut();

                            let first_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), pair.first.into()
                            ]).expect("missing gradient sample");
                            let second_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), pair.second.into()
                            ]).expect("missing gradient sample");

                            let array = gradient.values.to_array_mut();

                            for (property_i, &[distance]) in gradient.properties.iter_fixed_size().enumerate() {
                                if distance == 1 {
                                    array[[first_grad_sample_i, 0, 0, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 1, 1, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 2, 2, property_i]] = -1.0;

                                    array[[second_grad_sample_i, 0, 0, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 1, 1, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 2, 2, property_i]] = 1.0;
                                }
                            }
                        }
                    }
                }

                // then the pair second -> first
                if let Some(second_block_i) = second_block_i {
                    let mut block = descriptor.block_mut_by_id(second_block_i);

                    let block_data = block.data_mut();
                    let sample_i = block_data.samples.position(&[
                        LabelValue::from(system_i),
                        LabelValue::from(pair.second),
                        LabelValue::from(pair.first),
                        LabelValue::from(-cell_a),
                        LabelValue::from(-cell_b),
                        LabelValue::from(-cell_c),
                    ]);

                    if let Some(sample_i) = sample_i {
                        let array = block_data.values.to_array_mut();
                        for (property_i, &[distance]) in block_data.properties.iter_fixed_size().enumerate() {
                            if distance == 1 {
                                array[[sample_i, 0, property_i]] = -pair.vector[0];
                                array[[sample_i, 1, property_i]] = -pair.vector[1];
                                array[[sample_i, 2, property_i]] = -pair.vector[2];
                            }
                        }

                        if let Some(mut gradient) = block.gradient_mut("positions") {
                            let gradient = gradient.data_mut();

                            let first_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), pair.second.into()
                            ]).expect("missing gradient sample");
                            let second_grad_sample_i = gradient.samples.position(&[
                                sample_i.into(), system_i.into(), pair.first.into()
                            ]).expect("missing gradient sample");

                            let array = gradient.values.to_array_mut();

                            for (property_i, &[distance]) in gradient.properties.iter_fixed_size().enumerate() {
                                if distance == 1 {
                                    array[[first_grad_sample_i, 0, 0, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 1, 1, property_i]] = -1.0;
                                    array[[first_grad_sample_i, 2, 2, property_i]] = -1.0;

                                    array[[second_grad_sample_i, 0, 0, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 1, 1, property_i]] = 1.0;
                                    array[[second_grad_sample_i, 2, 2, property_i]] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use metatensor::Labels;

    use crate::systems::test_utils::{test_systems, test_system};
    use crate::Calculator;

    use super::NeighborList;
    use super::super::CalculatorBase;

    #[test]
    fn half_neighbor_list() {
        let mut calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 2.0,
            full_neighbor_list: false,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(*descriptor.keys(), Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[-42, 1], [1, 1]]
        ));

        // O-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_xyz"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // we have two O-H pairs
            &[[0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.75545], [-0.58895]],
            [[0.0], [-0.75545], [-0.58895]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-H block
        let block = descriptor.block_by_id(1);
        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // we have one H-H pair in the main image, and one more with
            // periodic images
            &[[0, 1, 2, 0, 0, 0], [0, 1, 2, 0, 1, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-1.5109], [0.0]],
            [[0.0], [1.4891], [0.0]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    #[test]
    fn full_neighbor_list() {
        let mut calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 2.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(*descriptor.keys(), Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[-42, 1], [1, -42], [1, 1]]
        ));

        // O-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_xyz"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // we have two O-H pairs
            &[[0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.75545], [-0.58895]],
            [[0.0], [-0.75545], [-0.58895]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-O block
        let block = descriptor.block_by_id(1);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_xyz"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // we have two H-O pairs
            &[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-0.75545], [0.58895]],
            [[0.0], [0.75545], [0.58895]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-H block
        let block = descriptor.block_by_id(2);
        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // we have one H-H pair, four times (including with periodic images)
            &[
                [0, 1, 2, 0, 0, 0], [0, 2, 1, 0, 0, 0],
                [0, 1, 2, 0, 1, 0], [0, 2, 1, 0, -1, 0],
            ]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-1.5109], [0.0]],
            [[0.0], [1.5109], [0.0]],
            [[0.0], [1.4891], [0.0]],
            [[0.0], [-1.4891], [0.0]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    #[test]
    fn periodic_neighbor_list() {
        let mut calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 12.0,
            full_neighbor_list: false,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["CH"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();
        assert_eq!(*descriptor.keys(), Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[1, 1], [1, 6], [6, 6]]
        ));

        // H-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // the pairs only differ in cell shifts
            &[[0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.0], [10.0]],
            [[0.0], [10.0], [0.0]],
            [[10.0], [0.0], [0.0]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // now a full NL
        let mut calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 12.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();
        assert_eq!(*descriptor.keys(), Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[1, 1], [1, 6], [6, 1], [6, 6]]
        ));

        // H-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.samples(), Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            // twice as many pairs
            &[
                [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, -1],
                [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, -1, 0],
                [0, 1, 1, 1, 0, 0], [0, 1, 1, -1, 0, 0],
            ]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.0], [10.0]],
            [[0.0], [0.0], [-10.0]],
            [[0.0], [10.0], [0.0]],
            [[0.0], [-10.0], [0.0]],
            [[10.0], [0.0], [0.0]],
            [[-10.0], [0.0], [0.0]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    #[test]
    fn finite_differences_positions() {
        // half neighbor list
        let calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 1.0,
            full_neighbor_list: false,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-9,
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);

        // full neighbor list
        let calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 1.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        // half neighbor list
        let calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 3.0,
            full_neighbor_list: false,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water", "methane"]);

        let samples = Labels::new(
            ["system", "first_atom"],
            &[[0, 1]],
        );

        let properties = Labels::new(
            ["distance"],
            &[[1]],
        );

        let keys = Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[-42, 1], [1, -42], [1, 1], [1, 6], [6, 1], [6, 6]]
        );

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );

        // full neighbor list
        let calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 3.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn check_self_pairs() {
        let mut calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 2.0,
            full_neighbor_list: true,
            self_pairs: true,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        // we have a block for O-O pairs (-42, -42)
        assert_eq!(descriptor.keys(), &Labels::new(
            ["first_atom_type", "second_atom_type"],
            &[[-42, -42], [-42, 1], [1, -42], [1, 1]]
        ));

        // H-H block
        let block = descriptor.block_by_id(3);
        let block = block.data();
        assert_eq!(*block.samples, Labels::new(
            ["system", "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            &[
                // we have two H-H pair in the main cell, two self-pairs and two
                // pairs between different periodic images.
                [0, 1, 2, 0, 0, 0],
                [0, 2, 1, 0, 0, 0],
                [0, 1, 2, 0, 1, 0],
                [0, 2, 1, 0, -1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 2, 2, 0, 0, 0],
            ]
        ));

    }
    #[test]
    fn check_empty_response() {
        let mut calculator = Calculator::from(Box::new(NeighborList {
            cutoff: 0.1,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        // cutoff too low, TensorMap is empty!
        assert_eq!(descriptor.keys(), &Labels::new::<i32,2>(
            ["first_atom_type", "second_atom_type"],
            &[],
        ));
    }
}
