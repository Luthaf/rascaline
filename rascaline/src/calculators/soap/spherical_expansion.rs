use std::cell::RefCell;
use std::sync::Arc;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use thread_local::ThreadLocal;

use ndarray::s;
use rayon::prelude::*;

use equistore::{LabelsBuilder, Labels, LabelValue, TensorBlockRefMut};
use equistore::TensorMap;

use crate::{Error, System, Vector3D, Matrix3};
use crate::systems::CellShape;

use crate::labels::{SamplesBuilder, SpeciesFilter, AtomCenteredSamples};
use crate::labels::{KeysBuilder, CenterSingleNeighborsSpeciesKeys};

use crate::math::SphericalHarmonicsCache;

use super::super::CalculatorBase;
use super::SoapRadialIntegralCache;

use super::radial_integral::SoapRadialIntegralParameters;
use super::{CutoffFunction, RadialScaling};
use crate::calculators::radial_basis::RadialBasis;

use super::super::{split_tensor_map_by_system, array_mut_for_system};

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// Parameters for spherical expansion calculator.
///
/// The spherical expansion is at the core of representations in the SOAP
/// (Smooth Overlap of Atomic Positions) family. See [this review
/// article](https://doi.org/10.1063/1.5090481) for more information on the SOAP
/// representation, and [this paper](https://doi.org/10.1063/5.0044689) for
/// information on how it is implemented in rascaline.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct SphericalExpansionParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,
    /// Weight of the central atom contribution to the
    /// features. If `1` the center atom contribution is weighted the same
    /// as any other contribution. If `0` the central atom does not
    /// contribute to the features at all.
    pub center_atom_weight: f64,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    #[serde(default)]
    pub radial_scaling: RadialScaling,
}

/// The actual calculator used to compute SOAP spherical expansion coefficients
pub struct SphericalExpansion {
    /// Parameters governing the spherical expansion
    parameters: SphericalExpansionParameters,
    /// implementation + cached allocation to compute the radial integral for a
    /// single pair
    radial_integral: ThreadLocal<RefCell<SoapRadialIntegralCache>>,
    /// implementation + cached allocation to compute the spherical harmonics
    /// for a single pair
    spherical_harmonics: ThreadLocal<RefCell<SphericalHarmonicsCache>>,
    /// Cache for (-1)^l values
    m_1_pow_l: Vec<f64>,
}

impl std::fmt::Debug for SphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SphericalExpansion {
    /// Create a new `SphericalExpansion` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansion, Error> {
        // validate parameters once in the constructor
        parameters.cutoff_function.validate()?;
        parameters.radial_scaling.validate()?;
        SoapRadialIntegralCache::new(parameters.radial_basis, SoapRadialIntegralParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            atomic_gaussian_width: parameters.atomic_gaussian_width,
            cutoff: parameters.cutoff,
        })?;

        let m_1_pow_l = (0..=parameters.max_angular).into_iter()
            .map(|l| f64::powi(-1.0, l as i32))
            .collect::<Vec<f64>>();

        return Ok(SphericalExpansion {
            parameters,
            radial_integral: ThreadLocal::new(),
            spherical_harmonics: ThreadLocal::new(),
            m_1_pow_l,
        });
    }

    /// Compute the product of radial scaling & cutoff smoothing functions
    fn scaling_functions(&self, r: f64) -> f64 {
        let cutoff = self.parameters.cutoff_function.compute(r, self.parameters.cutoff);
        let scaling = self.parameters.radial_scaling.compute(r);
        return cutoff * scaling;
    }

    /// Compute the gradient of the product of radial scaling & cutoff smoothing functions
    fn scaling_functions_gradient(&self, r: f64) -> f64 {
        let cutoff = self.parameters.cutoff_function.compute(r, self.parameters.cutoff);
        let cutoff_grad = self.parameters.cutoff_function.derivative(r, self.parameters.cutoff);

        let scaling = self.parameters.radial_scaling.compute(r);
        let scaling_grad = self.parameters.radial_scaling.derivative(r);

        return cutoff_grad * scaling + cutoff * scaling_grad;
    }

    /// Compute and add the self contribution to the spherical expansion
    /// coefficients, i.e. the contribution arising from the density of the
    /// center atom around itself.
    ///
    /// By symmetry, the center atom only contributes to the l=0 coefficients.
    /// It also does not have contributions to the gradients
    fn do_self_contributions(&mut self, systems: &[Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        debug_assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);
        // we could cache the self contribution since they only depend on the
        // gaussian atomic width. For now, we recompute them all the time

        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis,
                SoapRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                }
            ).expect("invalid parameters");
            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsCache::new(self.parameters.max_angular))
        }).borrow_mut();

        for (key, mut block) in descriptor.iter_mut() {
            let spherical_harmonics_l = key[0];
            let species_center = key[1];
            let species_neighbor = key[2];

            if spherical_harmonics_l != 0 || species_center != species_neighbor {
                // center contribution is non-zero only for l=0
                continue;
            }

            let values = block.values_mut();
            let array = values.data.as_array_mut();

            // Compute the three factors that appear in the center contribution.
            // Note that this is simply the pair contribution for the special
            // case where the pair distance is zero.
            radial_integral.compute(0.0, false);
            spherical_harmonics.compute(Vector3D::new(0.0, 0.0, 1.0), false);
            let f_scaling = self.scaling_functions(0.0);

            // The global factor of 4PI is used for the pair contributions as
            // well. See the relevant comments there for more details.
            let factor = self.parameters.center_atom_weight
                * FOUR_PI
                * f_scaling
                * spherical_harmonics.values[[0, 0]];

            // Add the center contribution to relevant elements of array.
            for (sample_i, &[structure, center]) in values.samples.iter_fixed_size().enumerate() {
                // it is possible that the samples from values.samples are not
                // part of the systems (the user requested extra samples). In
                // that case, we need to skip anything that does not exist, or
                // with a different species center
                if structure.usize() >= systems.len() {
                    continue;
                }

                let system = &systems[structure.usize()];
                if center.usize() > system.size()? {
                    continue;
                }

                if system.species()?[center.usize()] != species_center {
                    continue;
                }

                for (property_i, &[n]) in values.properties.iter_fixed_size().enumerate() {
                    array[[sample_i, 0, property_i]] += factor * radial_integral.values[[0, n.usize()]];
                }
            }
        }

        return Ok(());
    }

    /// Compute the contribution of a single pair and store the corresponding
    /// data inside the given descriptor.
    ///
    /// This will store data both for the spherical expansion with `pair.first`
    /// as the center and `pair.second` as the neighbor, and for the spherical
    /// expansion with `pair.second` as the center and `pair.first` as the
    /// neighbor.
    fn compute_for_pair(
        &self,
        distance: f64,
        direction: Vector3D,
        do_gradients: GradientsOptions,
        contribution: &mut PairContribution,
    ) {
        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis,
                SoapRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                }
            ).expect("invalid parameters");
            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsCache::new(self.parameters.max_angular))
        }).borrow_mut();

        radial_integral.compute(distance, do_gradients.either());
        spherical_harmonics.compute(direction, do_gradients.either());

        let f_scaling = self.scaling_functions(distance);
        let f_scaling_grad = self.scaling_functions_gradient(distance);

        let mut lm_index = 0;
        let mut lm_index_grad = 0;
        for spherical_harmonics_l in 0..=self.parameters.max_angular {
            let spherical_harmonics_grad = [
                spherical_harmonics.gradients[0].slice(spherical_harmonics_l as isize),
                spherical_harmonics.gradients[1].slice(spherical_harmonics_l as isize),
                spherical_harmonics.gradients[2].slice(spherical_harmonics_l as isize),
            ];
            let spherical_harmonics = spherical_harmonics.values.slice(spherical_harmonics_l as isize);

            let radial_integral_grad = radial_integral.gradients.slice(s![spherical_harmonics_l, ..]);
            let radial_integral = radial_integral.values.slice(s![spherical_harmonics_l, ..]);

            // compute the full spherical expansion coefficients & gradients
            for sph_value in spherical_harmonics.iter() {
                for (n, ri_value) in radial_integral.iter().enumerate() {
                    // The first factor of 4pi arises from the integration over
                    // the angular variables. It is included here as a global
                    // factor since it is not part of the spherical harmonics,
                    // and to keep the radial_integral class about the radial
                    // part of the integration only.
                    contribution.values[[lm_index, n]] = FOUR_PI * f_scaling * sph_value * ri_value;
                }
                lm_index += 1;
            }

            if let Some(ref mut gradient) = contribution.gradients {
                let dr_d_spatial = direction;

                for m in 0..(2 * spherical_harmonics_l + 1) {
                    let sph_value = spherical_harmonics[m];
                    let sph_grad_x = spherical_harmonics_grad[0][m];
                    let sph_grad_y = spherical_harmonics_grad[1][m];
                    let sph_grad_z = spherical_harmonics_grad[2][m];

                    for n in 0..self.parameters.max_radial {
                        let ri_value = radial_integral[n];
                        let ri_grad = radial_integral_grad[n];

                        gradient[[0, lm_index_grad, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[0] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[0] * sph_value
                            + f_scaling * ri_value * sph_grad_x / distance
                        );

                        gradient[[1, lm_index_grad, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[1] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[1] * sph_value
                            + f_scaling * ri_value * sph_grad_y / distance
                        );

                        gradient[[2, lm_index_grad, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[2] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[2] * sph_value
                            + f_scaling * ri_value * sph_grad_z / distance
                        );
                    }

                    lm_index_grad += 1;
                }
            }
        }
    }

    /// For one system, compute the spherical expansion and corresponding
    /// gradients by summing over the pairs.
    #[allow(clippy::too_many_lines)]
    fn accumulate_all_pairs(
        &self,
        system: &dyn System,
        do_gradients: GradientsOptions,
        requested_centers: &BTreeSet<usize>,
    ) -> Result<PairAccumulationResult, Error> {
        // pre-filter pairs to only include the ones containing at least one of
        // the requested atoms
        let pairs = system.pairs()?;

        let pair_should_contribute = |pair: &&crate::systems::Pair| {
            requested_centers.contains(&pair.first) || requested_centers.contains(&pair.second)
        };
        let pairs_count = pairs.iter().filter(pair_should_contribute).count();

        let system_size = system.size()?;
        let species = system.species()?;

        let mut species_mapping = BTreeMap::new();
        for &s in species {
            let next_idx = species_mapping.len();
            species_mapping.entry(s).or_insert(next_idx);
        }

        let inverse_cell = if do_gradients.cell {
            let cell = system.cell()?;
            if cell.shape() == CellShape::Infinite {
                return Err(Error::InvalidParameter(
                    "can not compute cell gradients for non periodic systems".into()
                ));
            }
            cell.matrix().inverse()
        } else {
            Matrix3::zero()
        };

        let centers_mapping = (0..system_size).map(|center_i| {
            requested_centers.iter().position(|&center| center == center_i)
        }).collect::<Vec<_>>();

        // total number of joined (l, m) indices
        let lm_shape = (self.parameters.max_angular + 1) * (self.parameters.max_angular + 1);
        let mut contribution = PairContribution {
            values: ndarray::Array2::from_elem((lm_shape, self.parameters.max_radial), 0.0),
            gradients: if do_gradients.either() {
                let shape = (3, lm_shape, self.parameters.max_radial);
                Some(ndarray::Array3::from_elem(shape, 0.0))
            } else {
                None
            }
        };

        let max_radial = self.parameters.max_radial;
        let mut result = PairAccumulationResult {
            values: ndarray::Array4::from_elem(
                (species_mapping.len(), requested_centers.len(), lm_shape, max_radial),
                0.0
            ),
            positions_gradients_by_pair: if do_gradients.positions {
                let shape = (pairs_count, 3, lm_shape, max_radial);
                Some(ndarray::Array4::from_elem(shape, 0.0))
            } else {
                None
            },
            positions_gradients_self: if do_gradients.positions {
                let shape = (species_mapping.len(), requested_centers.len(), 3, lm_shape, max_radial);
                Some(ndarray::Array5::from_elem(shape, 0.0))
            } else {
                None
            },
            cell_gradients: if do_gradients.cell {
                Some(ndarray::Array6::from_elem(
                    (species_mapping.len(), requested_centers.len(), 3, 3, lm_shape, max_radial),
                    0.0)
                )
            } else {
                None
            },
            species_mapping,
            centers_mapping,
            pair_to_pair_ids: HashMap::new(),
        };

        for (pair_id, pair) in pairs.iter().filter(pair_should_contribute).enumerate() {
            debug_assert!(requested_centers.contains(&pair.first) || requested_centers.contains(&pair.second));

            let mut direction = pair.vector / pair.distance;
            // Deal with the possibility that two atoms are at the same
            // position. While this is not usual, there is no reason to prevent
            // the calculation of spherical expansion. The user will still get a
            // warning about atoms being very close together when calculating
            // the neighbor list.
            if pair.distance < 1e-6 {
                direction = Vector3D::new(0.0, 0.0, 1.0);
            }

            self.compute_for_pair(pair.distance, direction, do_gradients, &mut contribution);

            let inverse_cell_pair_vector = Vector3D::new(
                pair.vector[0] * inverse_cell[0][0] + pair.vector[1] * inverse_cell[1][0] + pair.vector[2] * inverse_cell[2][0],
                pair.vector[0] * inverse_cell[0][1] + pair.vector[1] * inverse_cell[1][1] + pair.vector[2] * inverse_cell[2][1],
                pair.vector[0] * inverse_cell[0][2] + pair.vector[1] * inverse_cell[1][2] + pair.vector[2] * inverse_cell[2][2],
            );

            if let Some(mapped_center) = result.centers_mapping[pair.first] {
                // add the pair contribution to the atomic environnement
                // corresponding to the **first** atom in the pair
                let neighbor_i = pair.second;

                result.pair_to_pair_ids.entry((pair.first, pair.second))
                    .or_insert_with(Vec::new)
                    .push(pair_id);

                let species_neighbor_i = result.species_mapping[&species[neighbor_i]];
                let mut values = result.values.slice_mut(s![species_neighbor_i, mapped_center, .., ..]);
                values += &contribution.values;


                if let Some(ref contribution_gradients) = contribution.gradients {
                    if let Some(ref mut positions_gradients) = result.positions_gradients_by_pair {
                        let gradients = &mut positions_gradients.slice_mut(s![pair_id, .., .., ..]);
                        gradients.assign(contribution_gradients);
                    }

                    if let Some(ref mut positions_gradients) = result.positions_gradients_self {
                        let mut gradients = positions_gradients.slice_mut(s![species_neighbor_i, mapped_center, .., .., ..]);
                        gradients -= contribution_gradients;
                    }

                    if let Some(ref mut cell_gradients) = result.cell_gradients {
                        let mut cell_gradients = cell_gradients.slice_mut(
                            s![species_neighbor_i, mapped_center, .., .., .., ..]
                        );

                        for spatial_1 in 0..3 {
                            for spatial_2 in 0..3 {
                                let inverse_cell_pair_vector_2 = inverse_cell_pair_vector[spatial_2];

                                let mut lm_index = 0;
                                for spherical_harmonics_l in 0..=self.parameters.max_angular {
                                    for _m in 0..(2 * spherical_harmonics_l + 1) {
                                        for n in 0..self.parameters.max_radial {
                                            // SAFETY: we are doing in-bounds
                                            // access, and removing the bounds
                                            // checks is a significant speed-up
                                            // for this code. There is also a
                                            // bounds check when running tests
                                            // in debug mode.
                                            unsafe {
                                                let out = cell_gradients.uget_mut([spatial_1, spatial_2, lm_index, n]);
                                                *out += inverse_cell_pair_vector_2 * contribution_gradients.uget([spatial_1, lm_index, n]);
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

            if pair.first == pair.second {
                // do not compute for the reversed pair if the pair is
                // between an atom and its image
                continue;
            }

            if let Some(mapped_center) = result.centers_mapping[pair.second] {
                // add the pair contribution to the atomic environnement
                // corresponding to the **second** atom in the pair
                let neighbor_i = pair.first;

                result.pair_to_pair_ids.entry((pair.second, pair.first))
                    .or_insert_with(Vec::new)
                    .push(pair_id);


                // inverting the pair is equivalent to adding a (-1)^l
                // factor to the pair contribution values, and -(-1)^l
                // to the gradients
                let mut lm_index = 0;
                for spherical_harmonics_l in 0..=self.parameters.max_angular {
                    let factor = self.m_1_pow_l[spherical_harmonics_l];
                    for _m in 0..(2 * spherical_harmonics_l + 1) {
                        for n in 0..self.parameters.max_radial {
                            contribution.values[[lm_index, n]] *= factor;
                        }
                        lm_index += 1;
                    }
                }

                if let Some(ref mut gradients) = contribution.gradients {
                    for spatial in 0..3 {
                        let mut lm_index = 0;
                        for spherical_harmonics_l in 0..=self.parameters.max_angular {
                            let factor = -self.m_1_pow_l[spherical_harmonics_l];
                            for _m in 0..(2 * spherical_harmonics_l + 1) {
                                for n in 0..self.parameters.max_radial {
                                    gradients[[spatial, lm_index, n]] *= factor;
                                }
                                lm_index += 1;
                            }
                        }
                    }
                }

                let species_neighbor_i = result.species_mapping[&species[neighbor_i]];

                let mut values = result.values.slice_mut(s![species_neighbor_i, mapped_center, .., ..]);
                values += &contribution.values;

                if let Some(ref contribution_gradients) = contribution.gradients {
                    // we don't add second->first pair to positions_gradient_by_pair,
                    // instead handling this in position_gradients_to_equistore

                    if let Some(ref mut positions_gradients) = result.positions_gradients_self {
                        let mut gradients = positions_gradients.slice_mut(s![species_neighbor_i, mapped_center, .., .., ..]);
                        gradients -= contribution_gradients;
                    }

                    if let Some(ref mut cell_gradients) = result.cell_gradients {
                        let inverse_cell_pair_vector = -inverse_cell_pair_vector;

                        let mut cell_gradients = cell_gradients.slice_mut(
                            s![species_neighbor_i, mapped_center, .., .., .., ..]
                        );

                        for spatial_1 in 0..3 {
                            for spatial_2 in 0..3 {
                                let inverse_cell_pair_vector_2 = inverse_cell_pair_vector[spatial_2];

                                let mut lm_index = 0;
                                for spherical_harmonics_l in 0..=self.parameters.max_angular {
                                    for _m in 0..(2 * spherical_harmonics_l + 1) {
                                        for n in 0..self.parameters.max_radial {
                                            // SAFETY: as above
                                            unsafe {
                                                let out = cell_gradients.uget_mut([spatial_1, spatial_2, lm_index, n]);
                                                *out += inverse_cell_pair_vector_2 * contribution_gradients.uget([spatial_1, lm_index, n]);
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

    /// Move the pre-computed spherical expansion data to a single equistore
    /// block
    #[allow(clippy::unused_self)]
    fn values_to_equistore(
        &self,
        key: &[LabelValue],
        block: &mut TensorBlockRefMut,
        system: &dyn System,
        data: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let species = system.species()?;
        let system_size = system.size()?;

        let spherical_harmonics_l = key[0].usize();
        let species_center = key[1];
        let species_neighbor = key[2];

        let lm_start = spherical_harmonics_l * spherical_harmonics_l;
        let species_neighbor_i = if let Some(s) = data.species_mapping.get(&species_neighbor.i32()) {
            *s
        } else {
            // this block does not correspond to actual species in the current
            // system
            return Ok(());
        };

        let values = block.values_mut();
        let mut array = array_mut_for_system(&mut values.data);

        for (sample_i, [_, center_i]) in values.samples.iter_fixed_size().enumerate() {
            // samples might contain entries for atoms that should not be part
            // of this block, these entries can be manually requested by users.
            if center_i.usize() >= system_size || species[center_i.usize()] != species_center {
                continue;
            }
            let mapped_center = data.centers_mapping[center_i.usize()].expect("this center should be part of the mapping");

            for m in 0..(2 * spherical_harmonics_l + 1) {
                for (property_i, [n]) in values.properties.iter_fixed_size().enumerate() {
                    // SAFETY: we are doing in-bounds access, and removing the
                    // bounds checks is a significant speed-up for this code.
                    // There is also a bounds check when running tests in debug
                    // mode.
                    unsafe {
                        let out = array.uget_mut([sample_i, m, property_i]);
                        *out += *data.values.uget([species_neighbor_i, mapped_center, lm_start + m, n.usize()]);
                    }
                }
            }
        }

        return Ok(());
    }

    /// Finalize the spherical expansion gradients w.r.t. positions from the
    /// gradients associated with each pair, filling a single equistore block
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
    fn position_gradients_to_equistore(
        &self,
        key: &[LabelValue],
        block: &mut TensorBlockRefMut,
        system: &dyn System,
        data: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let positions_gradients_by_pair = if let Some(ref data) = data.positions_gradients_by_pair {
            data
        } else {
            // no positions gradients, return early
            return Ok(());
        };

        let positions_gradients_self = data.positions_gradients_self.as_ref()
            .expect("missing self gradients");

        let spherical_harmonics_l = key[0].usize();
        let species_center = key[1];
        let species_neighbor = key[2];
        let species_neighbor_i = if let Some(s) = data.species_mapping.get(&species_neighbor.i32()) {
            *s
        } else {
            // this block does not correspond to actual species in the current
            // system
            return Ok(());
        };

        let species = system.species()?;
        let pairs = system.pairs()?;
        let system_size = system.size()?;

        let lm_start = spherical_harmonics_l * spherical_harmonics_l;
        let m_1_pow_l = self.m_1_pow_l[spherical_harmonics_l];

        let values_samples = Arc::clone(&block.values().samples);
        let gradient = block.gradient_mut("positions").expect("TODO");
        let mut array = array_mut_for_system(&mut gradient.data);

        for (grad_sample_i, &[sample_i, _, neighbor_i]) in gradient.samples.iter_fixed_size().enumerate() {
            let center_i = values_samples[sample_i.usize()][1];

            // gradient samples should NOT contain entries for atoms that should
            // not be part of this block, since they are not manually specified
            // by the users
            debug_assert!(center_i.usize() < system_size && species[center_i.usize()] == species_center);

            if center_i == neighbor_i {
                // gradient of an environment w.r.t. the position of the center,
                // we already summed over the contributions from all pairs this
                // center is part of in `data.positions_gradients_self`

                let mapped_center = data.centers_mapping[center_i.usize()]
                    .expect("this center should be part of the requested centers");

                for spatial in 0..3 {
                    for m in 0..(2 * spherical_harmonics_l + 1) {
                        for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: same as above
                            unsafe {
                                let out = array.uget_mut([grad_sample_i, spatial, m, property_i]);
                                *out = *positions_gradients_self.uget(
                                    [species_neighbor_i, mapped_center, spatial, lm_start + m, n.usize()]
                                );
                            }
                        }
                    }
                }
            } else {
                // gradient w.r.t. the position of a neighboring atom
                let neighbor_i = neighbor_i.usize();
                debug_assert!(species[neighbor_i] == species_neighbor);

                for &pair_id in &data.pair_to_pair_ids[&(center_i.usize(), neighbor_i)] {
                    let pair = pairs[pair_id];
                    let factor = if pair.first == center_i.usize() {
                        debug_assert_eq!(pair.second, neighbor_i);
                        1.0
                    } else {
                        debug_assert!(pair.second == center_i.usize());
                        debug_assert_eq!(pair.first, neighbor_i);
                        -m_1_pow_l
                    };

                    for spatial in 0..3 {
                        for m in 0..(2 * spherical_harmonics_l + 1) {
                            for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                                // SAFETY: same as above
                                unsafe {
                                    let out = array.uget_mut([grad_sample_i, spatial, m, property_i]);
                                    *out += factor * *positions_gradients_by_pair.uget([pair_id, spatial, lm_start + m, n.usize()]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    /// Move the pre-computed spherical expansion gradients w.r.t. cell to
    /// a single equistore block
    #[allow(clippy::unused_self)]
    fn cell_gradients_to_equistore(
        &self,
        key: &[LabelValue],
        block: &mut TensorBlockRefMut,
        system: &dyn System,
        data: &PairAccumulationResult,
    ) -> Result<(), Error> {
        let contributions = if let Some(ref data) = data.cell_gradients {
            data
        } else {
            // no cell gradients, return early
            return Ok(());
        };

        let species = system.species()?;
        let system_size = system.size()?;

        let spherical_harmonics_l = key[0].usize();
        let species_center = key[1];
        let species_neighbor = key[2];

        let lm_start = spherical_harmonics_l * spherical_harmonics_l;
        let species_neighbor_i = if let Some(s) = data.species_mapping.get(&species_neighbor.i32()) {
            *s
        } else {
            // this block does not correspond to actual species in the current
            // system
            return Ok(());
        };

        let values_samples = Arc::clone(&block.values().samples);
        let gradient = block.gradient_mut("cell").expect("TODO");
        let mut array = array_mut_for_system(&mut gradient.data);

        for (grad_sample_i, [sample_i]) in gradient.samples.iter_fixed_size().enumerate() {
            let center_i = values_samples[sample_i.usize()][1];

            // gradient samples should NOT contain entries for atoms that should
            // not be part of this block, since they are not manually specified
            // by the users
            debug_assert!(center_i.usize() < system_size && species[center_i.usize()] == species_center);
            let mapped_center = data.centers_mapping[center_i.usize()].expect("this center should be part of the mapping");

            for spatial_1 in 0..3 {
                for spatial_2 in 0..3 {
                    for m in 0..(2 * spherical_harmonics_l + 1) {
                        for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: same as above
                            unsafe {
                                let out = array.uget_mut([grad_sample_i, spatial_1, spatial_2, m, property_i]);
                                *out += *contributions.uget([species_neighbor_i, mapped_center, spatial_1, spatial_2, lm_start + m, n.usize()]);
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GradientsOptions {
    positions: bool,
    cell: bool,
}

impl GradientsOptions {
    pub fn either(self) -> bool {
        return self.positions || self.cell;
    }
}


/// Contribution of a single pair to the spherical expansion
struct PairContribution {
    values: ndarray::Array2<f64>,
    gradients: Option<ndarray::Array3<f64>>,
}

/// Result of `accumulate_all_pairs`, summing over all pairs in a system
struct PairAccumulationResult {
    /// values of the spherical expansion
    ///
    /// the shape is [species_neighbor, mapped_center, lm_index, n]
    values: ndarray::Array4<f64>,
    /// gradients w.r.t. positions associated with each pair used in the
    /// calculation. This is used for gradients of a given center representation
    /// with respect to one of the neighbors
    ///
    /// the shape is [pair_id, spatial, lm_index, n]
    positions_gradients_by_pair: Option<ndarray::Array4<f64>>,
    /// gradient of spherical expansion w.r.t. the position of the central atom
    ///
    /// this is separate from `positions_gradients_by_pair` because it can be
    /// summed while computing each pair contributions.
    ///
    /// the shape is [species_neighbor, mapped_center, spatial, lm_index, n]
    positions_gradients_self: Option<ndarray::Array5<f64>>,
    /// gradients of the spherical expansion w.r.t. cell
    ///
    /// the shape is [species_neighbor, mapped_center, spatial_1, spatial_2, lm_index, n]
    cell_gradients: Option<ndarray::Array6<f64>>,

    /// Mapping from the species to the first dimension of values/cell_gradients
    species_mapping: BTreeMap<i32, usize>,
    /// Mapping from the atomic index to the second dimension of values/cell_gradients
    centers_mapping: Vec<Option<usize>>,
    /// Mapping from couples of atoms to (potentially multiple) pair_id (first
    /// dimension of positions_gradients_by_pair).
    ///
    /// Two atoms can have more than one pair between them, so we need to be
    /// able store more than one pair id.
    pair_to_pair_ids: HashMap<(usize, usize), Vec<usize>>,
}

impl CalculatorBase for SphericalExpansion {
    fn name(&self) -> String {
        "spherical expansion".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let builder = CenterSingleNeighborsSpeciesKeys {
            cutoff: self.parameters.cutoff,
            self_pairs: true,
        };
        let keys = builder.keys(systems)?;

        let mut builder = LabelsBuilder::new(vec!["spherical_harmonics_l", "species_center", "species_neighbor"]);
        for &[species_center, species_neighbor] in keys.iter_fixed_size() {
            for spherical_harmonics_l in 0..=self.parameters.max_angular {
                builder.add(&[spherical_harmonics_l.into(), species_center, species_neighbor]);
            }
        }

        return Ok(builder.finish());
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);

        // only compute the samples once for each `species_center, species_neighbor`,
        // and re-use the results across `spherical_harmonics_l`.
        let mut samples_per_species = BTreeMap::new();
        for [_, species_center, species_neighbor] in keys.iter_fixed_size() {
            if samples_per_species.contains_key(&(species_center, species_neighbor)) {
                continue;
            }

            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                self_pairs: true,
            };

            samples_per_species.insert((species_center, species_neighbor), builder.samples(systems)?);
        }

        let mut result = Vec::new();
        for [_, species_center, species_neighbor] in keys.iter_fixed_size() {
            let samples = samples_per_species.get(
                &(species_center, species_neighbor)
            ).expect("missing samples");

            result.push(Arc::clone(samples));
        }

        return Ok(result);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        match parameter {
            "positions" => true,
            "cell" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([_, species_center, species_neighbor], samples) in keys.iter_fixed_size().zip(samples) {
            // TODO: we don't need to rebuild the gradient samples for different
            // spherical_harmonics_l
            let builder = AtomCenteredSamples {
                cutoff: self.parameters.cutoff,
                species_center: SpeciesFilter::Single(species_center.i32()),
                species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                self_pairs: true,
            };

            gradient_samples.push(builder.gradients_for(systems, samples)?);
        }

        return Ok(gradient_samples);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Arc<Labels>>> {
        assert_eq!(keys.names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);

        // only compute the components once for each `spherical_harmonics_l`,
        // and re-use the results across `species_center, species_neighbor`.
        let mut component_by_l = BTreeMap::new();
        for [spherical_harmonics_l, _, _] in keys.iter_fixed_size() {
            if component_by_l.contains_key(spherical_harmonics_l) {
                continue;
            }

            let mut component = LabelsBuilder::new(vec!["spherical_harmonics_m"]);
            for m in -spherical_harmonics_l.i32()..=spherical_harmonics_l.i32() {
                component.add(&[LabelValue::new(m)]);
            }

            let components = vec![Arc::new(component.finish())];
            component_by_l.insert(*spherical_harmonics_l, components);
        }

        let mut result = Vec::new();
        for [spherical_harmonics_l, _, _] in keys.iter_fixed_size() {
            let components = component_by_l.get(spherical_harmonics_l).expect("missing samples");
            result.push(components.clone());
        }
        return result;
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Arc<Labels>> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }
        let properties = Arc::new(properties.finish());

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);

        let do_gradients = GradientsOptions {
            positions: descriptor.block_by_id(0).gradient("positions").is_some(),
            cell: descriptor.block_by_id(0).gradient("cell").is_some(),
        };
        self.do_self_contributions(systems, descriptor)?;
        let mut descriptors_by_system = split_tensor_map_by_system(descriptor, systems.len());

        systems.par_iter_mut()
            .zip_eq(&mut descriptors_by_system)
            .try_for_each(|(system, descriptor)| {
                system.compute_neighbors(self.parameters.cutoff)?;
                let system = &**system;

                // we will only run the calculation on pairs where one of the
                // atom is part of the requested samples
                let requested_centers = descriptor.iter().flat_map(|(_, block)| {
                    block.values().samples.iter().map(|sample| sample[1].usize())
                }).collect::<BTreeSet<_>>();

                let accumulated = self.accumulate_all_pairs(
                    system,
                    do_gradients,
                    &requested_centers,
                )?;

                // all pairs are done, copy the data into equistore, handling
                // any property selection made by the user
                for (key, mut block) in descriptor.iter_mut() {
                    self.values_to_equistore(key, &mut block, system, &accumulated)?;
                    self.position_gradients_to_equistore(key, &mut block, system, &accumulated)?;
                    self.cell_gradients_to_equistore(key, &mut block, system, &accumulated)?;
                }

                Ok::<_, Error>(())
            })?;

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use ndarray::ArrayD;
    use equistore::{Labels, TensorBlock, EmptyArray, LabelsBuilder, TensorMap};

    use crate::systems::test_utils::{test_systems, test_system};
    use crate::{Calculator, CalculationOptions, LabelsSelection};
    use crate::calculators::CalculatorBase;

    use super::{SphericalExpansion, SphericalExpansionParameters};
    use super::{CutoffFunction, RadialScaling};
    use crate::calculators::radial_basis::RadialBasis;


    fn parameters() -> SphericalExpansionParameters {
        SphericalExpansionParameters {
            cutoff: 3.5,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
            radial_scaling: RadialScaling::Willatt2018 { scale: 1.5, rate: 0.8, exponent: 2},
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
            for species_center in [1, -42] {
                for species_neighbor in [1, -42] {
                    let block_i = descriptor.keys().position(&[
                        l.into(), species_center.into() , species_neighbor.into()
                    ]);
                    assert!(block_i.is_some());
                    let block = &descriptor.block_by_id(block_i.unwrap());
                    let array = block.values().data.as_array();
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
            epsilon: 1e-16,
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
            epsilon: 1e-16,
        };
        crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
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

        let samples = Labels::new(["structure", "center"], &[
            [0, 2],
            [0, 1],
        ]);

        let keys = Labels::new(["spherical_harmonics_l", "species_center", "species_neighbor"], &[
            [0, -42, -42],
            [0, 6, 1], // not part of the default keys
            [2, -42, -42],
            [1, -42, -42],
            [1, -42, 1],
            [1, 1, -42],
            [0, -42, 1],
            [2, -42, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, -42],
            [2, 1, -42],
            [2, 1, 1],
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
        // species_center key.
        let block = TensorBlock::new(
            EmptyArray::new(vec![3, 1]),
            Arc::new(Labels::new(["structure", "center"], &[[0, 0], [0, 1], [0, 2]])),
            vec![],
            Arc::new(Labels::single()),
        ).unwrap();

        let mut keys = LabelsBuilder::new(vec!["spherical_harmonics_l", "species_center", "species_neighbor"]);
        let mut blocks = Vec::new();
        for l in 0..(parameters().max_angular + 1) as isize {
            for species_center in [1, -42] {
                for species_neighbor in [1, -42] {
                    keys.add(&[l, species_center, species_neighbor]);
                    blocks.push(block.clone());
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
        assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.keys()[0], [0, -42, -42]);

        let block = descriptor.block_by_id(0);
        let values = block.values();

        // entries centered on H atoms should be zero
        assert_eq!(*values.samples, Labels::new(["structure", "center"], &[[0, 0], [0, 1], [0, 2]]));
        let array = values.data.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 1), ArrayD::from_elem(vec![1, 6], 0.0));
        assert_eq!(array.index_axis(ndarray::Axis(0), 2), ArrayD::from_elem(vec![1, 6], 0.0));

        // get the block for hydrogen
        assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.keys()[21], [0, 1, 1]);

        let block = descriptor.block_by_id(21);
        let values = block.values();

        // entries centered on O atoms should be zero
        assert_eq!(*values.samples, Labels::new(["structure", "center"], &[[0, 0], [0, 1], [0, 2]]));
        let array = values.data.as_array();
        assert_eq!(array.index_axis(ndarray::Axis(0), 0), ArrayD::from_elem(vec![1, 6], 0.0));
    }
}
