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
/// Self pairs (pairs between an atom and periodic copy itself) can appear when
/// the cutoff is larger than the cell under periodic boundary conditions. Self
/// pairs with a distance of 0 are not included in this calculator, even though
/// they are required when computing SOAP.
///
/// This sample produces a single property (`"distance"`) with three components
/// (`"pair_direction"`) containing the x, y, and z component of the vector from
/// the first atom in the pair to the second. In addition to the atom indexes,
/// the samples also contain a pair index, to be able to distinguish between
/// multiple pairs between the same atom (if the cutoff is larger than the
/// cell).
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
    /// with the distance 0. The `pair_id` of such pairs is set to -1.
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

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        if self.full_neighbor_list {
            FullNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.keys(systems)
        } else {
            HalfNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.keys(systems)
        }
    }

    fn samples_names(&self) -> Vec<&str> {
        return vec!["structure", "pair_id", "first_atom", "second_atom"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite());

        if self.full_neighbor_list {
            FullNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.samples(keys, systems)
        } else {
            HalfNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.samples(keys, systems)
        }
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            // TODO: add support for cell gradients
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, _keys: &Labels, samples: &[Labels], _systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for block_samples in samples {
            let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);
            for (sample_i, &[system_i, pair_id, first, second]) in block_samples.iter_fixed_size().enumerate() {
                // self pairs do not contribute to gradients
                if pair_id == -1 {
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
        let components = vec![Labels::new(["pair_direction"], &[[0], [1], [2]])];
        return vec![components; keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["distance"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        properties.add(&[LabelValue::new(0)]);
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "NeighborList::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        if self.full_neighbor_list {
            FullNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.compute(systems, descriptor)
        } else {
            HalfNeighborList { cutoff: self.cutoff, self_pairs: self.self_pairs }.compute(systems, descriptor)
        }
    }
}

/// Implementation of half neighbor list, only including pairs once (such that
/// `species_i <= species_j`)
#[derive(Debug, Clone)]
struct HalfNeighborList {
    cutoff: f64,
    self_pairs: bool
}

impl HalfNeighborList {
    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let mut all_species_pairs = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;

            let species = system.species()?;
            for pair in system.pairs()? {
                let (species_pair, _) = sort_pair((species[pair.first], species[pair.second]));
                all_species_pairs.insert(species_pair);
            }

            // make sure we have self-pairs keys even if the system does not
            // contain any neighbors with the same species
            if self.self_pairs {
                for &species in species {
                    all_species_pairs.insert((species, species));
                }
            }
        }

        let mut keys = LabelsBuilder::new(vec!["species_first_atom", "species_second_atom"]);
        for (first, second) in all_species_pairs {
            keys.add(&[first, second]);
        }

        return Ok(keys.finish());
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for [species_first, species_second] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(
                vec!["structure", "pair_id", "first_atom", "second_atom"]
            );
            for (system_i, system) in systems.iter_mut().enumerate() {
                system.compute_neighbors(self.cutoff)?;
                let species = system.species()?;

                for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                    let ((species_i, species_j), invert) = sort_pair((species[pair.first], species[pair.second]));
                    let (atom_i, atom_j) = if invert {
                        (pair.second, pair.first)
                    } else {
                        (pair.first, pair.second)
                    };

                    if species_i == species_first.i32() && species_j == species_second.i32() {
                        builder.add(&[system_i, pair_id, atom_i, atom_j]);
                    }
                }

                // handle self pairs
                if self.self_pairs && species_first == species_second {
                    for center_i in 0..system.size()? {
                        if species[center_i] == species_first.i32() {
                            builder.add(&[
                                system_i.into(),
                                // set pair_id as -1 for self pairs
                                LabelValue::new(-1),
                                center_i.into(),
                                center_i.into(),
                            ]);
                        }
                    }
                }
            }


            results.push(builder.finish());
        }

        return Ok(results);
    }

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                // Sort the species in the pair to ensure a canonical order of
                // the atoms in it. This guarantee that multiple call to this
                // calculator always returns pairs in the same order, even if
                // the underlying neighbor list implementation (which comes from
                // the systems) changes.
                //
                // The `invert` variable tells us if we need to invert the pair
                // vector or not.
                let ((species_i, species_j), invert) = sort_pair((species[pair.first], species[pair.second]));

                let pair_vector = if invert {
                    -pair.vector
                } else {
                    pair.vector
                };

                let (atom_i, atom_j) = if invert {
                    (pair.second, pair.first)
                } else {
                    (pair.first, pair.second)
                };

                let block_id = descriptor.keys().position(&[
                    species_i.into(), species_j.into()
                ]).expect("missing block");

                let mut block = descriptor.block_mut_by_id(block_id);
                let block_data = block.data_mut();

                let sample_i = block_data.samples.position(&[
                    system_i.into(), pair_id.into(), atom_i.into(), atom_j.into()
                ]);

                if let Some(sample_i) = sample_i {
                    let array = block_data.values.to_array_mut();

                    array[[sample_i, 0, 0]] = pair_vector[0];
                    array[[sample_i, 1, 0]] = pair_vector[1];
                    array[[sample_i, 2, 0]] = pair_vector[2];

                    if let Some(mut gradient) = block.gradient_mut("positions") {
                        let gradient = gradient.data_mut();

                        let first_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), atom_i.into()
                        ]).expect("missing gradient sample");
                        let second_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), atom_j.into()
                        ]).expect("missing gradient sample");

                        let array = gradient.values.to_array_mut();

                        array[[first_grad_sample_i, 0, 0, 0]] = -1.0;
                        array[[first_grad_sample_i, 1, 1, 0]] = -1.0;
                        array[[first_grad_sample_i, 2, 2, 0]] = -1.0;

                        array[[second_grad_sample_i, 0, 0, 0]] = 1.0;
                        array[[second_grad_sample_i, 1, 1, 0]] = 1.0;
                        array[[second_grad_sample_i, 2, 2, 0]] = 1.0;
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
    pub self_pairs: bool
}

impl FullNeighborList {
    /// Get the list of keys for these systems (list of pair species present in the systems)
    pub(crate) fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let mut all_species_pairs = BTreeSet::new();
        for system in systems {
            system.compute_neighbors(self.cutoff)?;

            let species = system.species()?;
            for pair in system.pairs()? {
                all_species_pairs.insert((species[pair.first], species[pair.second]));
                all_species_pairs.insert((species[pair.second], species[pair.first]));
            }

            // make sure we have self-pairs keys even if the system does not
            // contain any neighbors with the same species
            if self.self_pairs {
                for &species in species {
                    all_species_pairs.insert((species, species));
                }
            }
        }

        let mut keys = LabelsBuilder::new(vec!["species_first_atom", "species_second_atom"]);
        for (first, second) in all_species_pairs {
            keys.add(&[first, second]);
        }

        return Ok(keys.finish());
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for &[species_first, species_second] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(
                vec!["structure", "pair_id", "first_atom", "second_atom"]
            );
            for (system_i, system) in systems.iter_mut().enumerate() {
                system.compute_neighbors(self.cutoff)?;
                let species = system.species()?;

                for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                    if species_first == species_second {
                        // same species for both atoms in the pair
                        if species[pair.first] == species_first.i32() && species[pair.second] == species_second.i32() {
                            builder.add(&[system_i, pair_id, pair.first, pair.second]);
                            if pair.first != pair.second {
                                // do not duplicate self pairs
                                builder.add(&[system_i, pair_id, pair.second, pair.first]);
                            }
                        }
                    } else {
                        // different species
                        if species[pair.first] == species_first.i32() && species[pair.second] == species_second.i32() {
                            builder.add(&[system_i, pair_id, pair.first, pair.second]);
                        } else if species[pair.second] == species_first.i32() && species[pair.first] == species_second.i32() {
                            builder.add(&[system_i, pair_id, pair.second, pair.first]);
                        }
                    }
                }

                // handle self pairs
                if self.self_pairs && species_first == species_second {
                    for center_i in 0..system.size()? {
                        if species[center_i] == species_first.i32() {
                            builder.add(&[
                                system_i.into(),
                                // set pair_id as -1 for self pairs
                                LabelValue::new(-1),
                                center_i.into(),
                                center_i.into(),
                            ]);
                        }
                    }
                }
            }

            results.push(builder.finish());
        }

        return Ok(results);
    }

    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                let first_block_id = descriptor.keys().position(&[
                    species[pair.first].into(), species[pair.second].into()
                ]).expect("missing block");

                let second_block_id = if species[pair.first] == species[pair.second] {
                    None
                } else {
                    Some(descriptor.keys().position(&[
                        species[pair.second].into(), species[pair.first].into()
                    ]).expect("missing block"))
                };

                // first, the pair first -> second
                let mut block = descriptor.block_mut_by_id(first_block_id);
                let block_data = block.data_mut();

                let sample_i = block_data.samples.position(&[
                    system_i.into(), pair_id.into(), pair.first.into(), pair.second.into()
                ]);

                if let Some(sample_i) = sample_i {
                    let array = block_data.values.to_array_mut();

                    array[[sample_i, 0, 0]] = pair.vector[0];
                    array[[sample_i, 1, 0]] = pair.vector[1];
                    array[[sample_i, 2, 0]] = pair.vector[2];

                    if let Some(mut gradient) = block.gradient_mut("positions") {
                        let gradient = gradient.data_mut();

                        let first_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), pair.first.into()
                        ]).expect("missing gradient sample");
                        let second_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), pair.second.into()
                        ]).expect("missing gradient sample");

                        let array = gradient.values.to_array_mut();

                        array[[first_grad_sample_i, 0, 0, 0]] = -1.0;
                        array[[first_grad_sample_i, 1, 1, 0]] = -1.0;
                        array[[first_grad_sample_i, 2, 2, 0]] = -1.0;

                        array[[second_grad_sample_i, 0, 0, 0]] = 1.0;
                        array[[second_grad_sample_i, 1, 1, 0]] = 1.0;
                        array[[second_grad_sample_i, 2, 2, 0]] = 1.0;
                    }
                }

                // then the pair second -> first
                let mut block = if let Some(second_block_id) = second_block_id {
                    descriptor.block_mut_by_id(second_block_id)
                } else {
                    if pair.first == pair.second {
                        // do not duplicate self pairs
                        continue
                    }
                    // same species for both atoms in the pair, keep the same block
                    block
                };

                let block_data = block.data_mut();
                let sample_i = block_data.samples.position(&[
                    system_i.into(), pair_id.into(), pair.second.into(), pair.first.into()
                ]);

                if let Some(sample_i) = sample_i {
                    let array = block_data.values.to_array_mut();

                    array[[sample_i, 0, 0]] = -pair.vector[0];
                    array[[sample_i, 1, 0]] = -pair.vector[1];
                    array[[sample_i, 2, 0]] = -pair.vector[2];

                    if let Some(mut gradient) = block.gradient_mut("positions") {
                        let gradient = gradient.data_mut();

                        let first_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), pair.second.into()
                        ]).expect("missing gradient sample");
                        let second_grad_sample_i = gradient.samples.position(&[
                            sample_i.into(), system_i.into(), pair.first.into()
                        ]).expect("missing gradient sample");

                        let array = gradient.values.to_array_mut();

                        array[[first_grad_sample_i, 0, 0, 0]] = -1.0;
                        array[[first_grad_sample_i, 1, 1, 0]] = -1.0;
                        array[[first_grad_sample_i, 2, 2, 0]] = -1.0;

                        array[[second_grad_sample_i, 0, 0, 0]] = 1.0;
                        array[[second_grad_sample_i, 1, 1, 0]] = 1.0;
                        array[[second_grad_sample_i, 2, 2, 0]] = 1.0;
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
            ["species_first_atom", "species_second_atom"],
            &[[-42, 1], [1, 1]]
        ));

        // O-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[0]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have two O-H pairs
            &[[0, 0, 0, 1], [0, 1, 0, 2]]
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
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have one H-H pair
            &[[0, 2, 1, 2]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-1.5109], [0.0]]
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
            ["species_first_atom", "species_second_atom"],
            &[[-42, 1], [1, -42], [1, 1]]
        ));

        // O-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[0]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have two O-H pairs
            &[[0, 0, 0, 1], [0, 1, 0, 2]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.75545], [-0.58895]],
            [[0.0], [-0.75545], [-0.58895]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-O block
        let block = descriptor.block_by_id(1);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[0]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["pair_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have two H-O pairs
            &[[0, 0, 1, 0], [0, 1, 2, 0]]
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
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have one H-H pair, twice
            &[[0, 2, 1, 2], [0, 2, 2, 1]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-1.5109], [0.0]],
            [[0.0], [1.5109], [0.0]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    #[test]
    fn finite_differences_positions() {
        // half neighbor list
        let calculator = Calculator::from(Box::new(NeighborList{
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
        let calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 1.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        // half neighbor list
        let calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 1.0,
            full_neighbor_list: false,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water", "methane"]);

        let samples = Labels::new(
            ["structure", "first_atom"],
            &[[0, 1]],
        );

        let properties = Labels::new(
            ["distance"],
            &[[0]],
        );

        let keys = Labels::new(
            ["species_first_atom", "species_second_atom"],
            &[[-42, 1], [1, -42], [1, 1], [6, 6]]
        );

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );

        // full neighbor list
        let calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 1.0,
            full_neighbor_list: true,
            self_pairs: false,
        }) as Box<dyn CalculatorBase>);
        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn check_self_pairs() {
        let mut calculator = Calculator::from(Box::new(NeighborList{
            cutoff: 2.0,
            full_neighbor_list: true,
            self_pairs: true,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        // we have a block for O-O pairs (-42, -42)
        assert_eq!(descriptor.keys(), &Labels::new(
            ["species_first_atom", "species_second_atom"],
            &[[-42, -42], [-42, 1], [1, -42], [1, 1]]
        ));

        // H-H block
        let block = descriptor.block_by_id(3);
        let block = block.data();
        assert_eq!(*block.samples, Labels::new(
            ["structure", "pair_id", "first_atom", "second_atom"],
            // we have one H-H pair and two self-pairs
            &[[0, 2, 1, 2], [0, 2, 2, 1], [0, -1, 1, 1], [0, -1, 2, 2]]
        ));

    }
}
