use std::cell::RefCell;
use std::sync::Arc;
use std::collections::{BTreeMap, BTreeSet};

use thread_local::ThreadLocal;

use ndarray::{Array2, Array3, s};
use rayon::prelude::*;

use equistore::{LabelsBuilder, Labels, LabelValue};
use equistore::{TensorMap, TensorBlockRefMut};

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

use super::super::{split_tensor_map_by_system, array_mut_for_system, TensorMapView};

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

/// A single pair involved in the calculation of spherical expansion
struct Pair {
    /// Sample associated with the first atom in the pair (`[system, first_atom]`)
    first_sample: [LabelValue; 2],
    /// Sample associated with the second atom in the pair (`[system, second_atom]`)
    second_sample: [LabelValue; 2],
    /// species of the first atom in the pair
    first_species: i32,
    /// species of the second atom in the pair
    second_species: i32,
    /// distance between the atoms in the pair
    distance: f64,
    /// normalized direction vector from the first to the second atom
    direction: Vector3D,
    /// full vector from the first to the second atom
    vector: Vector3D,
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
    #[allow(clippy::too_many_lines)]
    fn compute_for_pair(
        &self,
        pair: &Pair,
        descriptor: &mut TensorMapView,
        do_gradients: GradientsOptions,
        inverse_cell: &Matrix3,
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

        radial_integral.compute(pair.distance, do_gradients.either());
        spherical_harmonics.compute(pair.direction, do_gradients.either());

        let f_scaling = self.scaling_functions(pair.distance);
        let f_scaling_grad = self.scaling_functions_gradient(pair.distance);

        // Cache allocation for coefficients & gradients
        let shape = (2 * self.parameters.max_angular + 1, self.parameters.max_radial);
        let mut coefficients = Array2::from_elem(shape, 0.0);
        let mut coefficients_grad = if do_gradients.either() {
            let shape = (3, 2 * self.parameters.max_angular + 1, self.parameters.max_radial);
            Some(Array3::from_elem(shape, 0.0))
        } else {
            None
        };

        let inverse_cell_pair_vector = Vector3D::new(
            pair.vector[0] * inverse_cell[0][0] + pair.vector[1] * inverse_cell[1][0] + pair.vector[2] * inverse_cell[2][0],
            pair.vector[0] * inverse_cell[0][1] + pair.vector[1] * inverse_cell[1][1] + pair.vector[2] * inverse_cell[2][1],
            pair.vector[0] * inverse_cell[0][2] + pair.vector[1] * inverse_cell[1][2] + pair.vector[2] * inverse_cell[2][2],
        );

        for spherical_harmonics_l in 0..=self.parameters.max_angular {
            let first_block_id = descriptor.keys().position(&[
                spherical_harmonics_l.into(),
                pair.first_species.into(),
                pair.second_species.into()
            ]);

            let second_block_id = if pair.first_sample == pair.second_sample {
                // do not compute for the reversed pair if the pair is
                // between an atom and its image
                None
            } else {
                descriptor.keys().position(&[
                    spherical_harmonics_l.into(),
                    pair.second_species.into(),
                    pair.first_species.into()
                ])
            };

            if first_block_id.is_none() && second_block_id.is_none() {
                continue;
            }

            let spherical_harmonics_grad = [
                spherical_harmonics.gradients[0].slice(spherical_harmonics_l as isize),
                spherical_harmonics.gradients[1].slice(spherical_harmonics_l as isize),
                spherical_harmonics.gradients[2].slice(spherical_harmonics_l as isize),
            ];
            let spherical_harmonics = spherical_harmonics.values.slice(spherical_harmonics_l as isize);

            let radial_integral_grad = radial_integral.gradients.slice(s![spherical_harmonics_l, ..]);
            let radial_integral = radial_integral.values.slice(s![spherical_harmonics_l, ..]);

            // compute the full spherical expansion coefficients & gradients
            for (m, sph_value) in spherical_harmonics.iter().enumerate() {
                for (n, ri_value) in radial_integral.iter().enumerate() {
                    // The first factor of 4pi arises from the integration over
                    // the angular variables. It is included here as a global
                    // factor since it is not part of the spherical harmonics,
                    // and to keep the radial_integral class about the radial
                    // part of the integration only.
                    coefficients[[m, n]] = FOUR_PI * f_scaling * sph_value * ri_value;
                }
            }

            if let Some(ref mut coefficients_grad) = coefficients_grad {
                let dr_d_spatial = pair.direction;

                for m in 0..(2 * spherical_harmonics_l + 1) {
                    let sph_value = spherical_harmonics[m];
                    let sph_grad_x = spherical_harmonics_grad[0][m];
                    let sph_grad_y = spherical_harmonics_grad[1][m];
                    let sph_grad_z = spherical_harmonics_grad[2][m];

                    for n in 0..self.parameters.max_radial {
                        let ri_value = radial_integral[n];
                        let ri_grad = radial_integral_grad[n];

                        coefficients_grad[[0, m, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[0] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[0] * sph_value
                            + f_scaling * ri_value * sph_grad_x / pair.distance
                        );

                        coefficients_grad[[1, m, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[1] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[1] * sph_value
                            + f_scaling * ri_value * sph_grad_y / pair.distance
                        );

                        coefficients_grad[[2, m, n]] = FOUR_PI * (
                            f_scaling_grad * dr_d_spatial[2] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[2] * sph_value
                            + f_scaling * ri_value * sph_grad_z / pair.distance
                        );
                    }
                }
            }

            if let Some(block_id) = first_block_id {
                SphericalExpansion::accumulate_in_block(
                    descriptor.block_mut_by_id(block_id),
                    spherical_harmonics_l,
                    (pair.first_sample, pair.second_sample),
                    inverse_cell_pair_vector,
                    &coefficients,
                    &coefficients_grad,
                    do_gradients,
                );
            }

            if let Some(block_id) = second_block_id {
                // inverting the pair is equivalent to adding a (-1)^l factor to
                // the pair contribution values, and -(-1)^l to the gradients
                coefficients *= self.m_1_pow_l[spherical_harmonics_l];
                if let Some(ref mut coefficients_grad) = coefficients_grad {
                    *coefficients_grad *= -self.m_1_pow_l[spherical_harmonics_l];
                }

                SphericalExpansion::accumulate_in_block(
                    descriptor.block_mut_by_id(block_id),
                    spherical_harmonics_l,
                    (pair.second_sample, pair.first_sample),
                    -inverse_cell_pair_vector,
                    &coefficients,
                    &coefficients_grad,
                    do_gradients,
                );
            }
        }
    }

    /// Store the data as required in block, dealing with the user requesting a
    /// subset of properties/samples
    fn accumulate_in_block(
        mut block: TensorBlockRefMut,
        spherical_harmonics_l: usize,
        (first_sample, second_sample): ([LabelValue; 2], [LabelValue; 2]),
        inverse_cell_pair_vector: Vector3D,
        pair_contribution: &Array2<f64>,
        pair_contribution_grad: &Option<Array3<f64>>,
        do_gradient: GradientsOptions,
    ) {

        let values = block.values_mut();
        let mut array = array_mut_for_system(&mut values.data);

        let first_sample_position = values.samples.position(&first_sample);
        if first_sample_position.is_none() {
            // nothing to do
            return;
        }
        let sample_i = first_sample_position.expect("we just checked");

        for m in 0..(2 * spherical_harmonics_l + 1) {
            for (property_i, &[n]) in values.properties.iter_fixed_size().enumerate() {
                // SAFETY: we are doing in-bounds access, and removing
                // the bounds checks is a significant speed-up for this code
                unsafe {
                    let out = array.uget_mut([sample_i, m, property_i]);
                    *out += *pair_contribution.uget([m, n.usize()]);
                }
            }
        }

        // accumulate gradients w.r.t. positions
        if do_gradient.positions {
            let pair_contribution_grad = pair_contribution_grad.as_ref().expect("missing pair contribution to gradients");
            let gradient = block.gradient_mut("positions").expect("missing gradient storage");
            let mut array = array_mut_for_system(&mut gradient.data);

            // gradient of the first sample with respect to the first atom
            let self_grad_sample_i = gradient.samples.position(&[
                // first_sample contains [system_i, first_atom]
                sample_i.into(), first_sample[0], first_sample[1]
            ]);

            // gradient of the first sample with respect to the second atom
            let other_grad_sample_i = gradient.samples.position(&[
                sample_i.into(), second_sample[0], second_sample[1]
            ]);

            if let Some(gradient_sample_i) = self_grad_sample_i {
                for spatial in 0..3 {
                    for m in 0..(2 * spherical_harmonics_l + 1) {
                        for (property_i, &[n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: we are doing in-bounds access, and
                            // removing the bounds checks is a significant
                            // speed-up for this code
                            unsafe {
                                let out = array.uget_mut([gradient_sample_i, spatial, m, property_i]);
                                *out -= *pair_contribution_grad.uget([spatial, m, n.usize()]);
                            }
                        }
                    }
                }
            }

            if let Some(gradient_sample_i) = other_grad_sample_i {
                for spatial in 0..3 {
                    for m in 0..(2 * spherical_harmonics_l + 1) {
                        for (property_i, &[n]) in gradient.properties.iter_fixed_size().enumerate() {
                            // SAFETY: we are doing in-bounds access, and
                            // removing the bounds checks is a significant
                            // speed-up for this code
                            unsafe {
                                let out = array.uget_mut([gradient_sample_i, spatial, m, property_i]);
                                *out += *pair_contribution_grad.uget([spatial, m, n.usize()]);
                            }
                        }
                    }
                }
            }
        }

        // accumulate gradients w.r.t. cell vector
        if do_gradient.cell {
            let pair_contribution_grad = pair_contribution_grad.as_ref().expect("missing pair contribution to gradients");
            let gradient = block.gradient_mut("cell").expect("missing gradient storage");
            let mut array = array_mut_for_system(&mut gradient.data);

            // gradient of the first sample with respect to the first atom
            let gradient_sample_i = gradient.samples.position(&[sample_i.into()]);

            if let Some(gradient_sample_i) = gradient_sample_i {
                for spatial_1 in 0..3 {
                    for spatial_2 in 0..3 {
                        let inverse_cell_pair_vector_2 = inverse_cell_pair_vector[spatial_2];
                        for m in 0..(2 * spherical_harmonics_l + 1) {
                            for (property_i, &[n]) in gradient.properties.iter_fixed_size().enumerate() {
                                // SAFETY: we are doing in-bounds access, and
                                // removing the bounds checks is a significant
                                // speed-up for this code
                                unsafe {
                                    let out = array.uget_mut([gradient_sample_i, spatial_1, spatial_2, m, property_i]);
                                    *out += *pair_contribution_grad.uget([spatial_1, m, n.usize()])
                                            * inverse_cell_pair_vector_2;
                                }
                            }
                        }
                    }
                }
            }
        }
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
            .enumerate()
            .try_for_each(|(system_i, (system, descriptor))| {
                system.compute_neighbors(self.parameters.cutoff)?;
                let species = system.species()?;
                let pairs = system.pairs()?;

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

                let requested_centers = descriptor.iter().flat_map(|(_, block)| {
                    block.values().samples.iter().map(|sample| sample[1].usize())
                }).collect::<BTreeSet<_>>();

                for pair in pairs {
                    if !requested_centers.contains(&pair.first) && !requested_centers.contains(&pair.second) {
                        continue;
                    }

                    let mut direction = pair.vector / pair.distance;
                    // Deal with the possibility that two atoms are at the same
                    // position. While this is not usual, there is no reason to
                    // prevent the calculation of spherical expansion. The user will
                    // still get a warning about atoms being very close together
                    // when calculating the neighbor list.
                    if pair.distance < 1e-6 {
                        direction = Vector3D::new(0.0, 0.0, 1.0);
                    }

                    let pair = Pair {
                        first_sample: [system_i.into(), pair.first.into()],
                        second_sample: [system_i.into(), pair.second.into()],
                        first_species: species[pair.first],
                        second_species: species[pair.second],
                        distance: pair.distance,
                        direction: direction,
                        vector: pair.vector,
                    };

                    self.compute_for_pair(&pair, descriptor, do_gradients, &inverse_cell);
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
                    let block = &descriptor.blocks()[block_i.unwrap()];
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
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [2],
        ]);

        let samples = Labels::new(["structure", "center"], &[
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 2],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &samples, &properties
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
