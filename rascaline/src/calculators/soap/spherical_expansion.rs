use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray::s;
use rayon::prelude::*;

use metatensor::{LabelsBuilder, Labels, LabelValue, TensorBlockRefMut};
use metatensor::TensorMap;

use crate::{Error, System};

use crate::labels::{SamplesBuilder, AtomicTypeFilter, AtomCenteredSamples};
use crate::labels::{KeysBuilder, CenterSingleNeighborsTypesKeys};

use super::super::CalculatorBase;

use super::{SphericalExpansionByPair, SphericalExpansionParameters};
use super::spherical_expansion_pair::{GradientsOptions, PairContribution};

use super::super::{split_tensor_map_by_system, array_mut_for_system};


/// The actual calculator used to compute SOAP spherical expansion coefficients
#[derive(Debug)]
pub struct SphericalExpansion {
    /// Underlying calculator, computing spherical expansion on pair at the time
    by_pair: SphericalExpansionByPair,
    /// Cache for (-1)^l values
    m_1_pow_l: Vec<f64>,
}

impl SphericalExpansion {
    /// Create a new `SphericalExpansion` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansion, Error> {
        let m_1_pow_l = (0..=parameters.max_angular)
            .map(|l| f64::powi(-1.0, l as i32))
            .collect::<Vec<f64>>();

        return Ok(SphericalExpansion {
            by_pair: SphericalExpansionByPair::new(parameters)?,
            m_1_pow_l,
        });
    }

    /// Accumulate the self contribution to the spherical expansion
    /// coefficients, i.e. the contribution arising from the density of the
    /// center atom around itself.
    fn do_self_contributions(&mut self, systems: &[System], descriptor: &mut TensorMap) -> Result<(), Error> {
        debug_assert_eq!(descriptor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);

        let self_contribution = self.by_pair.self_contribution();

        for (key, mut block) in descriptor {
            let o3_lambda = key[0];
            let center_type = key[2];
            let neighbor_type = key[3];

            if o3_lambda != 0 || center_type != neighbor_type {
                // center contribution is non-zero only for l=0
                continue;
            }

            let block = block.data_mut();
            let array = block.values.to_array_mut();

            // Add the center contribution to relevant elements of array.
            for (sample_i, &[system_i, atom_i]) in block.samples.iter_fixed_size().enumerate() {
                // it is possible that the samples from values.samples are not
                // part of the systems (the user requested extra samples). In
                // that case, we need to skip anything that does not exist, or
                // with a different atomic type for the center
                if system_i.usize() >= systems.len() {
                    continue;
                }

                let system = &systems[system_i.usize()];
                if atom_i.usize() > system.size()? {
                    continue;
                }

                if system.types()?[atom_i.usize()] != center_type {
                    continue;
                }

                for (property_i, &[n]) in block.properties.iter_fixed_size().enumerate() {
                    array[[sample_i, 0, property_i]] += self_contribution.values[[0, n.usize()]];
                }
            }
        }

        return Ok(());
    }

    /// For one system, compute the spherical expansion and corresponding
    /// gradients by summing over the pairs.
    #[allow(clippy::too_many_lines)]
    fn accumulate_all_pairs(
        &self,
        system: &System,
        do_gradients: GradientsOptions,
        requested_atoms: &BTreeSet<usize>,
    ) -> Result<PairAccumulationResult, Error> {
        // pre-filter pairs to only include the ones containing at least one of
        // the requested atoms
        let pairs = system.pairs()?;

        let pair_should_contribute = |pair: &&crate::systems::Pair| {
            requested_atoms.contains(&pair.first) || requested_atoms.contains(&pair.second)
        };
        let pairs_count = pairs.iter().filter(pair_should_contribute).count();

        let system_size = system.size()?;
        let types = system.types()?;

        let mut types_mapping = BTreeMap::new();
        for &atomic_type in types {
            let next_idx = types_mapping.len();
            types_mapping.entry(atomic_type).or_insert(next_idx);
        }

        let center_mapping = (0..system_size).map(|atom_i| {
            requested_atoms.iter().position(|&atom| atom == atom_i)
        }).collect::<Vec<_>>();


        let max_angular = self.by_pair.parameters().max_angular;
        let max_radial = self.by_pair.parameters().max_radial;
        let mut contribution = PairContribution::new(max_radial, max_angular, do_gradients.any());

        // total number of joined (l, m) indices
        let lm_shape = (max_angular + 1) * (max_angular + 1);
        let mut result = PairAccumulationResult {
            values: ndarray::Array4::from_elem(
                (types_mapping.len(), requested_atoms.len(), lm_shape, max_radial),
                0.0
            ),
            positions_gradient_by_pair: if do_gradients.positions {
                let shape = (pairs_count, 3, lm_shape, max_radial);
                Some(ndarray::Array4::from_elem(shape, 0.0))
            } else {
                None
            },
            self_positions_gradients: if do_gradients.positions {
                let shape = (types_mapping.len(), requested_atoms.len(), 3, lm_shape, max_radial);
                Some(ndarray::Array5::from_elem(shape, 0.0))
            } else {
                None
            },
            cell_gradients: if do_gradients.cell {
                Some(ndarray::Array6::from_elem(
                    (types_mapping.len(), requested_atoms.len(), 3, 3, lm_shape, max_radial),
                    0.0)
                )
            } else {
                None
            },
            strain_gradients: if do_gradients.strain {
                Some(ndarray::Array6::from_elem(
                    (types_mapping.len(), requested_atoms.len(), 3, 3, lm_shape, max_radial),
                    0.0)
                )
            } else {
                None
            },
            types_mapping,
            center_mapping,
            pairs_for_positions_gradient: HashMap::new(),
        };

        for (pair_id, pair) in pairs.iter().filter(pair_should_contribute).enumerate() {
            debug_assert!(requested_atoms.contains(&pair.first) || requested_atoms.contains(&pair.second));

            let direction = pair.vector / pair.distance;

            self.by_pair.compute_for_pair(pair.distance, direction, do_gradients, &mut contribution);

            if let Some(mapped_center) = result.center_mapping[pair.first] {
                // add the pair contribution to the atomic environnement
                // corresponding to the **first** atom in the pair
                let neighbor_i = pair.second;

                result.pairs_for_positions_gradient.entry((pair.first, pair.second))
                    .or_default()
                    .push(pair_id);

                let neighbor_type_i = result.types_mapping[&types[neighbor_i]];
                let mut values = result.values.slice_mut(s![neighbor_type_i, mapped_center, .., ..]);
                values += &contribution.values;


                if let Some(ref contribution_gradients) = contribution.gradients {
                    if let Some(ref mut positions_gradients) = result.positions_gradient_by_pair {
                        let gradients = &mut positions_gradients.slice_mut(s![pair_id, .., .., ..]);
                        gradients.assign(contribution_gradients);
                    }

                    if pair.first != pair.second {
                        if let Some(ref mut positions_gradients) = result.self_positions_gradients {
                            let mut gradients = positions_gradients.slice_mut(s![neighbor_type_i, mapped_center, .., .., ..]);
                            gradients -= contribution_gradients;
                        }
                    }

                    if let Some(ref mut cell_gradients) = result.cell_gradients {
                        let mut cell_gradients = cell_gradients.slice_mut(
                            s![neighbor_type_i, mapped_center, .., .., .., ..]
                        );

                        for abc in 0..3 {
                            let shift = pair.cell_shift_indices[abc] as f64;
                            for xyz in 0..3 {
                                let mut lm_index = 0;
                                for o3_lambda in 0..=max_angular {
                                    for _m in 0..(2 * o3_lambda + 1) {
                                        for n in 0..max_radial {
                                            // SAFETY: we are doing in-bounds access, and removing the bounds
                                            // checks is a significant speed-up for this code. The bounds are
                                            // still checked in debug mode
                                            unsafe {
                                                let out = cell_gradients.uget_mut([abc, xyz, lm_index, n]);
                                                *out += shift * contribution_gradients.uget([xyz, lm_index, n]);
                                            }
                                        }
                                        lm_index += 1;
                                    }
                                }
                            }
                        }
                    }

                    if let Some(ref mut strain_gradients) = result.strain_gradients {
                        let mut strain_gradients = strain_gradients.slice_mut(
                            s![neighbor_type_i, mapped_center, .., .., .., ..]
                        );

                        for xyz_1 in 0..3 {
                            for xyz_2 in 0..3 {
                                let mut lm_index = 0;
                                for o3_lambda in 0..=max_angular {
                                    for _m in 0..(2 * o3_lambda + 1) {
                                        for n in 0..max_radial {
                                            // SAFETY: same as above
                                            unsafe {
                                                let out = strain_gradients.uget_mut([xyz_1, xyz_2, lm_index, n]);
                                                *out += pair.vector[xyz_1] * contribution_gradients.uget([xyz_2, lm_index, n]);
                                            }
                                        }
                                        lm_index += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(mapped_center) = result.center_mapping[pair.second] {
                // add the pair contribution to the atomic environnement
                // corresponding to the **second** atom in the pair
                let neighbor_i = pair.first;

                result.pairs_for_positions_gradient.entry((pair.second, pair.first))
                    .or_default()
                    .push(pair_id);

                contribution.inverse_pair(&self.m_1_pow_l);

                let neighbor_type_i = result.types_mapping[&types[neighbor_i]];

                let mut values = result.values.slice_mut(s![neighbor_type_i, mapped_center, .., ..]);
                values += &contribution.values;

                if let Some(ref contribution_gradients) = contribution.gradients {
                    // we don't add second->first pair to positions_gradient_by_pair,
                    // instead handling this in position_gradients_to_metatensor

                    if pair.first != pair.second {
                        if let Some(ref mut positions_gradients) = result.self_positions_gradients {
                            let mut gradients = positions_gradients.slice_mut(s![neighbor_type_i, mapped_center, .., .., ..]);
                            gradients -= contribution_gradients;
                        }
                    }

                    if let Some(ref mut cell_gradients) = result.cell_gradients {
                        let mut cell_gradients = cell_gradients.slice_mut(
                            s![neighbor_type_i, mapped_center, .., .., .., ..]
                        );

                        for abc in 0..3 {
                            let shift = pair.cell_shift_indices[abc] as f64;
                            for xyz in 0..3 {
                                let mut lm_index = 0;
                                for o3_lambda in 0..=max_angular {
                                    for _m in 0..(2 * o3_lambda + 1) {
                                        for n in 0..max_radial {
                                            // SAFETY: we are doing in-bounds access, and removing the bounds
                                            // checks is a significant speed-up for this code. The bounds are
                                            // still checked in debug mode
                                            unsafe {
                                                let out = cell_gradients.uget_mut([abc, xyz, lm_index, n]);
                                                *out += -shift * contribution_gradients.uget([xyz, lm_index, n]);
                                            }
                                        }
                                        lm_index += 1;
                                    }
                                }
                            }
                        }
                    }

                    if let Some(ref mut strain_gradients) = result.strain_gradients {
                        let mut strain_gradients = strain_gradients.slice_mut(
                            s![neighbor_type_i, mapped_center, .., .., .., ..]
                        );

                        for xyz_1 in 0..3 {
                            for xyz_2 in 0..3 {
                                let mut lm_index = 0;
                                for o3_lambda in 0..=max_angular {
                                    for _m in 0..(2 * o3_lambda + 1) {
                                        for n in 0..max_radial {
                                            // SAFETY: as above
                                            unsafe {
                                                let out = strain_gradients.uget_mut([xyz_1, xyz_2, lm_index, n]);
                                                *out += -pair.vector[xyz_1] * contribution_gradients.uget([xyz_2, lm_index, n]);
                                            }
                                        }
                                        lm_index += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(result);
    }

    /// Move the pre-computed spherical expansion data to a single metatensor
    /// block
    #[allow(clippy::unused_self)]
    fn values_to_metatensor(
        &self,
        key: &[LabelValue],
        block: &mut TensorBlockRefMut,
        system: &System,
        result: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let types = system.types()?;
        let system_size = system.size()?;

        let o3_lambda = key[0].usize();
        let center_type = key[2];
        let neighbor_type = key[3];

        let lm_start = o3_lambda * o3_lambda;
        let neighbor_type_i = if let Some(s) = result.types_mapping.get(&neighbor_type.i32()) {
            *s
        } else {
            // this block does not correspond to actual types in the current system
            return Ok(());
        };

        let block = block.data_mut();
        let mut array = array_mut_for_system(block.values);

        for (sample_i, [_, atom_i]) in block.samples.iter_fixed_size().enumerate() {
            // samples might contain entries for atoms that should not be part
            // of this block, these entries can be manually requested by users.
            if atom_i.usize() >= system_size || types[atom_i.usize()] != center_type {
                continue;
            }
            let mapped_center = result.center_mapping[atom_i.usize()].expect("this atom should be part of the mapping");

            for m in 0..(2 * o3_lambda + 1) {
                for (property_i, [n]) in block.properties.iter_fixed_size().enumerate() {
                    // SAFETY: we are doing in-bounds access, and removing the
                    // bounds checks is a significant speed-up for this code.
                    // There is also a bounds check when running tests in debug
                    // mode.
                    unsafe {
                        let out = array.uget_mut([sample_i, m, property_i]);
                        *out += *result.values.uget([neighbor_type_i, mapped_center, lm_start + m, n.usize()]);
                    }
                }
            }
        }

        return Ok(());
    }

    /// Finalize the spherical expansion gradients w.r.t. positions from the
    /// gradients associated with each pair, filling a single metatensor block
    ///
    /// A single pair between atoms `i-j` will contribute to four gradients
    /// entries. Two "cross" gradients:
    ///     - gradient of the spherical expansion of `i` w.r.t. `j`;
    ///     - gradient of the spherical expansion of `j` w.r.t. `i`;
    /// and two "self" gradients:
    ///     - gradient of the spherical expansion of `i` w.r.t. `i`;
    ///     - gradient of the spherical expansion of `j` w.r.t. `j`;
    ///
    /// The self gradients are pre-summed in `accumulate_all_pairs`, while cross
    /// gradients are directly summed in this function.
    fn position_gradients_to_metatensor(
        &self,
        key: &[LabelValue],
        block: &mut TensorBlockRefMut,
        system: &System,
        result: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let positions_gradients = if let Some(ref data) = result.positions_gradient_by_pair {
            data
        } else {
            // no positions gradients, return early
            return Ok(());
        };

        let self_positions_gradients = result.self_positions_gradients.as_ref().expect("missing self gradients");

        let o3_lambda = key[0].usize();
        let center_type = key[2];
        let neighbor_type = key[3];
        let neighbor_type_i = if let Some(s) = result.types_mapping.get(&neighbor_type.i32()) {
            *s
        } else {
            // this block does not correspond to actual types in the current
            // system. It was requested by the user with samples/property
            // selection, but we don't need to do anything.
            return Ok(());
        };

        let types = system.types()?;
        let pairs = system.pairs()?;
        let system_size = system.size()?;

        let lm_start = o3_lambda * o3_lambda;
        let m_1_pow_l = self.m_1_pow_l[o3_lambda];

        let values_samples = block.samples();

        let mut gradient = block.gradient_mut("positions").expect("missing positions gradients");
        let gradient = gradient.data_mut();
        let mut array = array_mut_for_system(gradient.values);

        for (grad_sample_i, &[sample_i, _, neighbor_i]) in gradient.samples.iter_fixed_size().enumerate() {
            let center_i = values_samples[sample_i.usize()][1].usize();
            let neighbor_i = neighbor_i.usize();

            // gradient samples should NOT contain entries for atoms that should
            // not be part of this block, since they are not manually specified
            // by the users
            debug_assert!(center_i < system_size && types[center_i] == center_type);

            if center_i == neighbor_i {
                // gradient of an environment w.r.t. the position of the center
                let mapped_center = result.center_mapping[center_i]
                    .expect("this center should be part of the requested centers");

                for xyz in 0..3 {
                    for m in 0..(2 * o3_lambda + 1) {
                        for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: same as above
                            unsafe {
                                let out = array.uget_mut([grad_sample_i, xyz, m, property_i]);
                                *out = *self_positions_gradients.uget(
                                    [neighbor_type_i, mapped_center, xyz, lm_start + m, n.usize()]
                                );
                            }
                        }
                    }
                }
            } else {
                // gradient w.r.t. the position of a neighboring atom
                debug_assert!(types[neighbor_i] == neighbor_type);
                for &pair_id in &result.pairs_for_positions_gradient[&(center_i, neighbor_i)] {
                    let pair = pairs[pair_id];
                    let factor = if pair.first == center_i {
                        debug_assert_eq!(pair.second, neighbor_i);
                        1.0
                    } else {
                        debug_assert_eq!(pair.second, center_i);
                        debug_assert_eq!(pair.first, neighbor_i);
                        -m_1_pow_l
                    };

                    for xyz in 0..3 {
                        for m in 0..(2 * o3_lambda + 1) {
                            for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                                // SAFETY: same as above
                                unsafe {
                                    let out = array.uget_mut([grad_sample_i, xyz, m, property_i]);
                                    *out += factor * *positions_gradients.uget([pair_id, xyz, lm_start + m, n.usize()]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    /// Move the pre-computed spherical expansion gradients w.r.t. strain to
    /// a single metatensor block
    #[allow(clippy::unused_self)]
    fn cell_strain_gradients_to_metatensor(
        &self,
        key: &[LabelValue],
        parameter: &str,
        block: &mut TensorBlockRefMut,
        system: &System,
        result: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let contributions = if parameter == "strain" {
            if let Some(ref data) = result.strain_gradients {
                data
            } else {
                // no gradients, return early
                return Ok(());
            }
        } else if parameter == "cell" {
            if let Some(ref data) = result.cell_gradients {
                data
            } else {
                // no gradients, return early
                return Ok(());
            }
        } else {
            panic!("invalid gradient parameter: {}", parameter);
        };

        let types = system.types()?;
        let system_size = system.size()?;

        let o3_lambda = key[0].usize();
        let center_type = key[2];
        let neighbor_type = key[3];

        let lm_start = o3_lambda * o3_lambda;
        let neighbor_type_i = if let Some(s) = result.types_mapping.get(&neighbor_type.i32()) {
            *s
        } else {
            // this block does not correspond to actual types in the current system
            return Ok(());
        };

        let values_samples = block.samples();
        let mut gradient = block.gradient_mut(parameter).expect("missing gradients");
        let gradient = gradient.data_mut();
        let mut array = array_mut_for_system(gradient.values);

        for (grad_sample_i, [sample_i]) in gradient.samples.iter_fixed_size().enumerate() {
            let atom_i = values_samples[sample_i.usize()][1];

            if atom_i.usize() >= system_size || types[atom_i.usize()] != center_type {
                // the atom sample can be given by the user through sample
                // selection and not match an actual atom
                continue;
            }

            let mapped_center = result.center_mapping[atom_i.usize()].expect("this atom should be part of the mapping");

            for xyz_1 in 0..3 {
                for xyz_2 in 0..3 {
                    for m in 0..(2 * o3_lambda + 1) {
                        for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: same as above
                            unsafe {
                                let out = array.uget_mut([grad_sample_i, xyz_1, xyz_2, m, property_i]);
                                *out += *contributions.uget([neighbor_type_i, mapped_center, xyz_1, xyz_2, lm_start + m, n.usize()]);
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}

/// Result of `accumulate_all_pairs`, summing over all pairs in a system
struct PairAccumulationResult {
    /// values of the spherical expansion
    ///
    /// the shape is `[neighbor_type, mapped_center, lm_index, n]`
    values: ndarray::Array4<f64>,
    /// Gradients w.r.t. positions associated with each pair used in the
    /// calculation. This is used for gradients of a given center representation
    /// with respect to one of the neighbors
    ///
    /// the shape is `[pair_id, xyz, lm_index, n]`
    positions_gradient_by_pair: Option<ndarray::Array4<f64>>,
    /// gradient of spherical expansion w.r.t. the position of the central atom
    ///
    /// this is separate from `positions_gradient_by_pair` because it can be
    /// summed while computing each pair contributions.
    ///
    /// the shape is `[neighbor_types, mapped_center, xyz, lm_index, n]`
    self_positions_gradients: Option<ndarray::Array5<f64>>,
    /// gradients of the spherical expansion w.r.t. cell
    ///
    /// the shape is `[neighbor_type, mapped_center, xyz_1, xyz_2, lm_index, n]`
    cell_gradients: Option<ndarray::Array6<f64>>,
    /// gradients of the spherical expansion w.r.t. strain
    ///
    /// the shape is `[neighbor_type, mapped_center, xyz_1, xyz_2, lm_index, n]`
    strain_gradients: Option<ndarray::Array6<f64>>,

    /// Mapping from atomic types to the first dimension of values/cell
    /// gradients/strain gradients
    types_mapping: BTreeMap<i32, usize>,
    /// Mapping from the atomic index to the second dimension of
    /// values/cell gradients/strain gradients
    center_mapping: Vec<Option<usize>>,
    /// Mapping from (center, neighbor) to (potentially multiple) `pair_id`
    /// (first dimension of `positions_gradient_by_pair`).
    ///
    /// Two atoms can have more than one pair between them, so we need to be
    /// able store more than one pair id.
    pairs_for_positions_gradient: HashMap<(usize, usize), Vec<usize>>,
}

impl CalculatorBase for SphericalExpansion {
    fn name(&self) -> String {
        "spherical expansion".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self.by_pair.parameters()).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        self.by_pair.cutoffs()
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        let builder = CenterSingleNeighborsTypesKeys {
            cutoff: self.by_pair.parameters().cutoff,
            self_pairs: true,
        };
        let keys = builder.keys(systems)?;

        let mut builder = LabelsBuilder::new(vec!["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        for &[center_type, neighbor_type] in keys.iter_fixed_size() {
            for o3_lambda in 0..=self.by_pair.parameters().max_angular {
                builder.add(&[o3_lambda.into(), 1.into(), center_type, neighbor_type]);
            }
        }

        return Ok(builder.finish());
    }

    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);

        // only compute the samples once for each `center_type, neighbor_type`,
        // and re-use the results across `o3_lambda`.
        let mut samples_per_types = BTreeMap::new();
        for [_, _, center_type, neighbor_type] in keys.iter_fixed_size() {
            if samples_per_types.contains_key(&(center_type, neighbor_type)) {
                continue;
            }

            let builder = AtomCenteredSamples {
                cutoff: self.by_pair.parameters().cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: true,
            };

            samples_per_types.insert((center_type, neighbor_type), builder.samples(systems)?);
        }

        let mut result = Vec::new();
        for [_, _, center_type, neighbor_type] in keys.iter_fixed_size() {
            let samples = samples_per_types.get(
                &(center_type, neighbor_type)
            ).expect("missing samples");

            result.push(samples.clone());
        }

        return Ok(result);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" | "cell" | "strain" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([_, _, center_type, neighbor_type], samples) in keys.iter_fixed_size().zip(samples) {
            // TODO: we don't need to rebuild the gradient samples for different
            // o3_lambda
            let builder = AtomCenteredSamples {
                cutoff: self.by_pair.parameters().cutoff,
                center_type: AtomicTypeFilter::Single(center_type.i32()),
                neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                self_pairs: true,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        assert_eq!(keys.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);

        // only compute the components once for each `o3_lambda`,
        // and re-use the results across `center_type, neighbor_type`.
        let mut component_by_l = BTreeMap::new();
        for [o3_lambda, _, _, _] in keys.iter_fixed_size() {
            if component_by_l.contains_key(o3_lambda) {
                continue;
            }

            let mut component = LabelsBuilder::new(vec!["o3_mu"]);
            for m in -o3_lambda.i32()..=o3_lambda.i32() {
                component.add(&[LabelValue::new(m)]);
            }

            let components = vec![component.finish()];
            component_by_l.insert(*o3_lambda, components);
        }

        let mut result = Vec::new();
        for [o3_lambda, _, _, _] in keys.iter_fixed_size() {
            let components = component_by_l.get(o3_lambda).expect("missing samples");
            result.push(components.clone());
        }
        return result;
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for n in 0..self.by_pair.parameters().max_radial {
            properties.add(&[n]);
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert!(descriptor.keys().count() > 0);

        let do_gradients = GradientsOptions {
            positions: descriptor.block_by_id(0).gradient("positions").is_some(),
            strain: descriptor.block_by_id(0).gradient("strain").is_some(),
            cell: descriptor.block_by_id(0).gradient("cell").is_some(),
        };
        self.do_self_contributions(systems, descriptor)?;
        let mut descriptors_by_system = split_tensor_map_by_system(descriptor, systems.len());

        systems.par_iter_mut()
            .zip_eq(&mut descriptors_by_system)
            .try_for_each(|(system, descriptor)| {
                system.compute_neighbors(self.by_pair.parameters().cutoff)?;
                let system = &*system;

                // we will only run the calculation on pairs where one of the
                // atom is part of the requested samples
                let requested_centers = descriptor.iter().flat_map(|(_, block)| {
                    block.samples().iter().map(|sample| sample[1].usize()).collect::<Vec<_>>()
                }).collect::<BTreeSet<_>>();

                let accumulated = self.accumulate_all_pairs(
                    system,
                    do_gradients,
                    &requested_centers,
                )?;

                // all pairs are done, copy the data into metatensor, handling
                // any property selection made by the user
                for (key, mut block) in descriptor.iter_mut() {
                    self.values_to_metatensor(key, &mut block, system, &accumulated)?;
                    self.position_gradients_to_metatensor(key, &mut block, system, &accumulated)?;
                    self.cell_strain_gradients_to_metatensor(key, "cell", &mut block, system, &accumulated)?;
                    self.cell_strain_gradients_to_metatensor(key, "strain", &mut block, system, &accumulated)?;
                }

                Ok::<_, Error>(())
            })?;

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use ndarray::ArrayD;
    use metatensor::{Labels, TensorBlock, EmptyArray, LabelsBuilder, TensorMap};

    use crate::systems::test_utils::{test_systems, test_system};
    use crate::{Calculator, CalculationOptions, LabelsSelection};
    use crate::calculators::CalculatorBase;

    use super::{SphericalExpansion, SphericalExpansionParameters};
    use super::super::{CutoffFunction, RadialScaling};
    use crate::calculators::radial_basis::RadialBasis;


    fn parameters() -> SphericalExpansionParameters {
        SphericalExpansionParameters {
            cutoff: 7.3,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
            radial_scaling: RadialScaling::Willatt2018 { scale: 1.5, rate: 0.8, exponent: 2.0},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        for l in 0..6 {
            for center_type in [1, -42] {
                for neighbor_type in [1, -42] {
                    let block_i = descriptor.keys().position(&[
                        l.into(), 1.into(), center_type.into() , neighbor_type.into()
                    ]);
                    assert!(block_i.is_some());
                    let block = &descriptor.block_by_id(block_i.unwrap());
                    let array = block.values().to_array();
                    assert_eq!(array.shape().len(), 3);
                    assert_eq!(array.shape()[1], 2 * l + 1);
                }
            }
        }

        // exact values for spherical expansion are regression-tested in
        // `rascaline/tests/spherical-expansion.rs`
    }

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-9,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    }

    #[test]
    fn finite_differences_cell() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-9,
        };
        crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
    }

    #[test]
    fn finite_differences_strain() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-6,
            max_relative: 1e-5,
            epsilon: 1e-9,
        };
        crate::calculators::tests_utils::finite_differences_strain(calculator, &system, options);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            SphericalExpansionParameters {
                max_angular: 2,
                ..parameters()
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [2],
        ]);

        let samples = Labels::new(["system", "atom"], &[
            [0, 2],
            [0, 1],
        ]);

        let keys = Labels::new(["o3_lambda", "o3_sigma", "center_type", "neighbor_type"], &[
            [0, 1, -42, -42],
            [0, 1, 6, 1], // not part of the default keys
            [2, 1, -42, -42],
            [1, 1, -42, -42],
            [1, 1, -42, 1],
            [1, 1, 1, -42],
            [0, 1, -42, 1],
            [2, 1, -42, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, -42],
            [2, 1, 1, -42],
            [2, 1, 1, 1],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn non_existing_samples() {
        let mut calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        // include the three atoms in all blocks, regardless of the
        // center_type key.
        let block = TensorBlock::new(
            EmptyArray::new(vec![3, 1]),
            &Labels::new(["system", "atom"], &[[0, 0], [0, 1], [0, 2]]),
            &[],
            &Labels::single(),
        ).unwrap();

        let mut keys = LabelsBuilder::new(vec!["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        let mut blocks = Vec::new();
        for l in 0..(parameters().max_angular + 1) as isize {
            for center_type in [1, -42] {
                for neighbor_type in [1, -42] {
                    keys.add(&[l, 1, center_type, neighbor_type]);
                    blocks.push(block.as_ref().try_clone().unwrap());
                }
            }
        }
        let select_all_samples = TensorMap::new(keys.finish(), blocks).unwrap();

        let options = CalculationOptions {
            selected_samples: LabelsSelection::Predefined(&select_all_samples),
            ..Default::default()
        };
        let descriptor = calculator.compute(&mut systems, options).unwrap();

        // get the block for oxygen
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(descriptor.keys()[0], [0, 1, -42, -42]);

        let block = descriptor.block_by_id(0);
        let block = block.data();

        // entries centered on H atoms should be zero
        assert_eq!(*block.samples, Labels::new(["system", "atom"], &[[0, 0], [0, 1], [0, 2]]));
        let array = block.values.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 1), ArrayD::from_elem(vec![1, 6], 0.0));
        assert_eq!(array.index_axis(ndarray::Axis(0), 2), ArrayD::from_elem(vec![1, 6], 0.0));

        // get the block for hydrogen
        assert_eq!(descriptor.keys().names(), ["o3_lambda","o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(descriptor.keys()[21], [0, 1, 1, 1]);

        let block = descriptor.block_by_id(21);
        let block = block.data();

        // entries centered on O atoms should be zero
        assert_eq!(*block.samples, Labels::new(["system", "atom"], &[[0, 0], [0, 1], [0, 2]]));
        let array = block.values.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 0), ArrayD::from_elem(vec![1, 6], 0.0));
    }
}
