use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Entry;
use std::cell::RefCell;

use ndarray::s;
use thread_local::ThreadLocal;

use metatensor::{Labels, LabelsBuilder, LabelValue, TensorMap, TensorBlockRefMut};

use crate::{Error, System, Vector3D, Matrix3};
use crate::systems::CellShape;

use crate::math::SphericalHarmonicsCache;

use super::super::CalculatorBase;
use super::super::neighbor_list::FullNeighborList;

use super::{CutoffFunction, RadialScaling};

use crate::calculators::radial_basis::RadialBasis;
use super::SoapRadialIntegralCache;

use super::radial_integral::SoapRadialIntegralParameters;

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

impl SphericalExpansionParameters {
    /// Validate all the parameters
    pub fn validate(&self) -> Result<(), Error> {
        self.cutoff_function.validate()?;
        self.radial_scaling.validate()?;

        // try constructing a radial integral
        SoapRadialIntegralCache::new(self.radial_basis.clone(), SoapRadialIntegralParameters {
            max_radial: self.max_radial,
            max_angular: self.max_angular,
            atomic_gaussian_width: self.atomic_gaussian_width,
            cutoff: self.cutoff,
        })?;

        return Ok(());
    }
}

/// The actual calculator used to compute spherical expansion pair-by-pair
pub struct SphericalExpansionByPair {
    pub(crate) parameters: SphericalExpansionParameters,
    /// implementation + cached allocation to compute the radial integral for a
    /// single pair
    radial_integral: ThreadLocal<RefCell<SoapRadialIntegralCache>>,
    /// implementation + cached allocation to compute the spherical harmonics
    /// for a single pair
    spherical_harmonics: ThreadLocal<RefCell<SphericalHarmonicsCache>>,
    /// Cache for (-1)^l values
    m_1_pow_l: Vec<f64>,
}

impl std::fmt::Debug for SphericalExpansionByPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}


/// Which gradients are we computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GradientsOptions {
    pub positions: bool,
    pub cell: bool,
}

impl GradientsOptions {
    pub fn either(self) -> bool {
        return self.positions || self.cell;
    }
}


/// Contribution of a single pair to the spherical expansion
pub(super) struct PairContribution {
    /// Values of the contribution. The shape is (lm, n), where the lm index
    /// runs over both l and m
    pub values: ndarray::Array2<f64>,
    /// Gradients of the contribution w.r.t. the distance between the atoms in
    /// the pair. The shape is (x/y/z, lm, n).
    pub gradients: Option<ndarray::Array3<f64>>,
}

impl PairContribution {
    pub fn new(max_radial: usize, max_angular: usize, do_gradients: bool) -> PairContribution {
        let lm_shape = (max_angular + 1) * (max_angular + 1);
        PairContribution {
            values: ndarray::Array2::from_elem((lm_shape, max_radial), 0.0),
            gradients: if do_gradients {
                Some(ndarray::Array3::from_elem((3, lm_shape, max_radial), 0.0))
            } else {
                None
            }
        }
    }

    /// Modify the values/gradients as required to construct the
    /// values/gradients associated with pair j -> i from pair i -> j.
    ///
    /// `m_1_pow_l` should contain the values of `(-1)^l` up to `max_angular`
    pub fn inverse_pair(&mut self, m_1_pow_l: &[f64]) {
        let max_angular = m_1_pow_l.len() - 1;
        let max_radial = self.values.shape()[1];
        debug_assert_eq!(self.values.shape()[0], (max_angular + 1) * (max_angular + 1));

        // inverting the pair is equivalent to adding a (-1)^l factor to the
        // pair contribution values, and -(-1)^l to the gradients
        let mut lm_index = 0;
        for spherical_harmonics_l in 0..=max_angular {
            let factor = m_1_pow_l[spherical_harmonics_l];
            for _m in 0..(2 * spherical_harmonics_l + 1) {
                for n in 0..max_radial {
                    self.values[[lm_index, n]] *= factor;
                }
                lm_index += 1;
            }
        }

        if let Some(ref mut gradients) = self.gradients {
            for spatial in 0..3 {
                let mut lm_index = 0;
                for spherical_harmonics_l in 0..=max_angular {
                    let factor = -m_1_pow_l[spherical_harmonics_l];
                    for _m in 0..(2 * spherical_harmonics_l + 1) {
                        for n in 0..max_radial {
                            gradients[[spatial, lm_index, n]] *= factor;
                        }
                        lm_index += 1;
                    }
                }
            }
        }
    }
}


impl SphericalExpansionByPair {
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansionByPair, Error> {
        parameters.validate()?;

        let m_1_pow_l = (0..=parameters.max_angular)
            .map(|l| f64::powi(-1.0, l as i32))
            .collect::<Vec<f64>>();

        Ok(SphericalExpansionByPair {
            parameters: parameters,
            radial_integral: ThreadLocal::new(),
            spherical_harmonics: ThreadLocal::new(),
            m_1_pow_l,
        })
    }

    /// Access the spherical expansion parameters used by this calculator
    pub fn parameters(&self) -> &SphericalExpansionParameters {
        &self.parameters
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

    /// Compute the self-contribution (contribution coming from an atom "seeing"
    /// it's own density). This is equivalent to a normal pair contribution,
    /// with a distance of 0.
    ///
    /// For now, the same density is used for all atoms, so this function can be
    /// called only once and re-used for all atoms (see `do_self_contributions`
    /// below).
    ///
    /// By symmetry, the self-contribution is only non-zero for `L=0`, and does
    /// not contributes to the gradients.
    pub(super) fn self_contribution(&self) -> PairContribution {
        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                SoapRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                }
            ).expect("invalid radial integral parameters");
            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsCache::new(self.parameters.max_angular))
        }).borrow_mut();

        // Compute the three factors that appear in the center contribution.
        // Note that this is simply the pair contribution for the special
        // case where the pair distance is zero.
        radial_integral.compute(0.0, false);
        spherical_harmonics.compute(Vector3D::new(0.0, 0.0, 1.0), false);
        let f_scaling = self.scaling_functions(0.0);

        let factor = self.parameters.center_atom_weight
            * f_scaling
            * spherical_harmonics.values[[0, 0]];

        radial_integral.values *= factor;

        return PairContribution {
            values: radial_integral.values.clone(),
            gradients: None
        };
    }

    /// Accumulate the self contribution to the spherical expansion
    /// coefficients.
    ///
    /// For the pair-by-pair spherical expansion, we use a special `pair_id`
    /// (-1) to store the data associated with self-pairs.
    fn do_self_contributions(&self, systems: &[Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        debug_assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_atom_1", "species_atom_2"]);
        let self_contribution = self.self_contribution();

        for (key, mut block) in descriptor {
            let spherical_harmonics_l = key[0];
            let species_atom_1 = key[1];
            let species_atom_2 = key[2];

            if spherical_harmonics_l != 0 || species_atom_1 != species_atom_2 {
                // center contribution is non-zero only for l=0
                continue;
            }

            let data = block.data_mut();
            let array = data.values.to_array_mut();

            // loop over all samples in this block, find self pairs
            // (`pair_id` is -1), and fill the data using `self_contribution`
            for (sample_i, &[structure, pair_id, atom_1, atom_2]) in data.samples.iter_fixed_size().enumerate() {
                // it is possible that the samples from values.samples are not
                // part of the systems (the user requested extra samples). In
                // that case, we need to skip anything that does not exist, or
                // with a different species center
                if structure.usize() >= systems.len() || pair_id != -1 {
                    continue;
                }

                let system = &systems[structure.usize()];
                let n_atoms = system.size()?;
                let species = system.species()?;

                if atom_1.usize() > n_atoms || atom_2.usize() > n_atoms {
                    continue;
                }

                if species[atom_1.usize()] != species_atom_1 || species[atom_2.usize()] != species_atom_2 {
                    continue;
                }

                for (property_i, &[n]) in data.properties.iter_fixed_size().enumerate() {
                    array[[sample_i, 0, property_i]] = self_contribution.values[[0, n.usize()]];
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
    pub(super) fn compute_for_pair(
        &self,
        distance: f64,
        mut direction: Vector3D,
        do_gradients: GradientsOptions,
        contribution: &mut PairContribution,
    ) {
        debug_assert!(distance >= 0.0);

        // Deal with the possibility that two atoms are at the same
        // position. While this is not usual, there is no reason to
        // prevent the calculation of spherical expansion. The user will
        // still get a warning about atoms being very close together
        // when calculating the neighbor list.
        if distance < 1e-6 {
            direction = Vector3D::new(0.0, 0.0, 1.0);
        }

        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
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
            for sph_value in spherical_harmonics {
                for (n, ri_value) in radial_integral.iter().enumerate() {
                    contribution.values[[lm_index, n]] = f_scaling * sph_value * ri_value;
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

                        gradient[[0, lm_index_grad, n]] =
                            f_scaling_grad * dr_d_spatial[0] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[0] * sph_value
                            + f_scaling * ri_value * sph_grad_x / distance;

                        gradient[[1, lm_index_grad, n]] =
                            f_scaling_grad * dr_d_spatial[1] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[1] * sph_value
                            + f_scaling * ri_value * sph_grad_y / distance;

                        gradient[[2, lm_index_grad, n]] =
                            f_scaling_grad * dr_d_spatial[2] * ri_value * sph_value
                            + f_scaling * ri_grad * dr_d_spatial[2] * sph_value
                            + f_scaling * ri_value * sph_grad_z / distance;
                    }

                    lm_index_grad += 1;
                }
            }
        }
    }

    /// Accumulate a single pair `contribution` in the right block.
    fn accumulate_in_block(
        spherical_harmonics_l: usize,
        mut block: TensorBlockRefMut,
        sample: &[LabelValue],
        contribution: &PairContribution,
        do_gradients: GradientsOptions,
        inverse_cell_pair_vector: Vector3D,
    ) {
        let data = block.data_mut();
        let array = data.values.to_array_mut();

        let sample_i = data.samples.position(sample);

        if let Some(sample_i) = sample_i {
            let lm_start = spherical_harmonics_l * spherical_harmonics_l;

            for m in 0..(2 * spherical_harmonics_l + 1) {
                for (property_i, [n]) in data.properties.iter_fixed_size().enumerate() {
                    unsafe {
                        let out = array.uget_mut([sample_i, m, property_i]);
                        *out += *contribution.values.uget([lm_start + m, n.usize()]);
                    }
                }
            }

            if let Some(ref contribution_gradients) = contribution.gradients {
                if do_gradients.positions {
                    let mut gradient = block.gradient_mut("positions").expect("missing positions gradients");
                    let gradient = gradient.data_mut();

                    let array = gradient.values.to_array_mut();
                    debug_assert_eq!(gradient.samples.names(), ["sample", "structure", "atom"]);

                    // gradient of the pair contribution w.r.t. the position of
                    // the first atom
                    let first_grad_sample_i = gradient.samples.position(&[
                        sample_i.into(), /* structure */ sample[0], /* pair.first */ sample[2]
                    ]).expect("missing first gradient sample");

                    for spatial in 0..3 {
                        for m in 0..(2 * spherical_harmonics_l + 1) {
                            for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                                unsafe {
                                    let out = array.uget_mut([first_grad_sample_i, spatial, m, property_i]);
                                    *out -= contribution_gradients.uget([spatial, lm_start + m, n.usize()]);
                                }
                            }
                        }
                    }

                    // gradient of the pair contribution w.r.t. the position of
                    // the second atom
                    let second_grad_sample_i = gradient.samples.position(&[
                        sample_i.into(), /* structure */ sample[0], /* pair.second */ sample[3]
                    ]).expect("missing second gradient sample");

                    for spatial in 0..3 {
                        for m in 0..(2 * spherical_harmonics_l + 1) {
                            for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                                unsafe {
                                    let out = array.uget_mut([second_grad_sample_i, spatial, m, property_i]);
                                    *out += contribution_gradients.uget([spatial, lm_start + m, n.usize()]);
                                }
                            }
                        }
                    }
                }

                if do_gradients.cell {
                    let mut gradient = block.gradient_mut("cell").expect("missing cell gradients");
                    let gradient = gradient.data_mut();

                    debug_assert_eq!(gradient.samples.names(), ["sample"]);
                    assert_eq!(gradient.samples[sample_i][0].usize(), sample_i);

                    let array = gradient.values.to_array_mut();
                    for spatial_1 in 0..3 {
                        for spatial_2 in 0..3 {
                            let inverse_cell_pair_vector_2 = inverse_cell_pair_vector[spatial_2];

                            for m in 0..(2 * spherical_harmonics_l + 1) {
                                for (property_i, [n]) in gradient.properties.iter_fixed_size().enumerate() {
                                    unsafe {
                                        let out = array.uget_mut([sample_i, spatial_1, spatial_2, m, property_i]);
                                        *out += inverse_cell_pair_vector_2 * contribution_gradients.uget([spatial_1, lm_start + m, n.usize()]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


impl CalculatorBase for SphericalExpansionByPair {
    fn name(&self) -> String {
        "spherical expansion by pair".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        std::slice::from_ref(&self.parameters.cutoff)
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        // the species part of the keys is the same for all l
        let species_keys = FullNeighborList { cutoff: self.parameters.cutoff, self_pairs: false }.keys(systems)?;
        let mut all_species_pairs = species_keys.iter().map(|p| (p[0], p[1])).collect::<BTreeSet<_>>();

        // also include self-pairs in case they are missing from species_keys
        let mut all_species = BTreeSet::new();
        for system in systems {
            let species = system.species()?;
            for &species in species {
                all_species.insert(species);
            }
        }

        for species in all_species {
            all_species_pairs.insert((species.into(), species.into()));
        }

        let mut keys = LabelsBuilder::new(vec![
            "spherical_harmonics_l",
            "species_atom_1",
            "species_atom_2"
        ]);

        for (s1, s2) in all_species_pairs {
            for l in 0..=self.parameters.max_angular {
                keys.add(&[l.into(), s1, s2]);
            }
        }


        return Ok(keys.finish());
    }

    fn samples_names(&self) -> Vec<&str> {
        return vec!["structure", "pair_id", "first_atom", "second_atom"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        let mut result = Vec::new();

        // we only need to compute samples once for each l, the cache stores the
        // ones we have already computed
        let mut cache: BTreeMap<_, Labels> = BTreeMap::new();
        for &[_, s1, s2] in keys.iter_fixed_size() {
            let samples = match cache.entry((s1, s2)) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    let mut builder = LabelsBuilder::new(self.samples_names());

                    for (system_i, system) in systems.iter_mut().enumerate() {
                        system.compute_neighbors(self.parameters.cutoff)?;
                        let species = system.species()?;

                        if s1 == s2 {
                            // same species for the two atoms, we need to insert
                            // the self pairs
                            for center_i in 0..system.size()? {
                                if species[center_i] == s1.i32() {
                                    builder.add(&[
                                        system_i.into(),
                                        // set pair_id as -1 for self pairs
                                        LabelValue::new(-1),
                                        center_i.into(),
                                        center_i.into(),
                                    ]);
                                }
                            }

                            for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                                if species[pair.first] == s1.i32() && species[pair.second] == s2.i32() {
                                    builder.add(&[system_i, pair_id, pair.first, pair.second]);
                                    if pair.first != pair.second {
                                        // do not duplicate actual pairs between
                                        // an atom and itself (these can exist
                                        // when the cutoff is larger than the
                                        // cell in periodic boundary conditions)
                                        builder.add(&[system_i, pair_id, pair.second, pair.first]);
                                    }
                                }
                            }
                        } else {
                            // different species for the two atoms
                            for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                                if species[pair.first] == s1.i32() && species[pair.second] == s2.i32() {
                                    builder.add(&[system_i, pair_id, pair.first, pair.second]);
                                } else if species[pair.second] == s1.i32() && species[pair.first] == s2.i32() {
                                    builder.add(&[system_i, pair_id, pair.second, pair.first]);
                                }
                            }
                        }
                    }

                    let samples = builder.finish();
                    entry.insert(samples).clone()
                }
            };

            result.push(samples);
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

    fn positions_gradient_samples(&self, _: &Labels, samples: &[Labels], _: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
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
        assert_eq!(keys.names().len(), 3);
        assert_eq!(keys.names()[0], "spherical_harmonics_l");

        let mut result = Vec::new();
        // only compute the components once for each `spherical_harmonics_l`,
        // and re-use the results across the other keys.
        let mut cache: BTreeMap<_, Vec<Labels>> = BTreeMap::new();
        for &[spherical_harmonics_l, _, _] in keys.iter_fixed_size() {
            let components = match cache.entry(spherical_harmonics_l) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    let mut component = LabelsBuilder::new(vec!["spherical_harmonics_m"]);
                    for m in -spherical_harmonics_l.i32()..=spherical_harmonics_l.i32() {
                        component.add(&[LabelValue::new(m)]);
                    }

                    let components = vec![component.finish()];
                    entry.insert(components).clone()
                }
            };

            result.push(components);
        }

        return result;
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }

        return vec![properties.finish(); keys.count()];
    }

    #[time_graph::instrument(name = "SphericalExpansionByPair::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_atom_1", "species_atom_2"]);

        let do_gradients = GradientsOptions {
            positions: descriptor.block_by_id(0).gradient("positions").is_some(),
            cell: descriptor.block_by_id(0).gradient("cell").is_some(),
        };

        self.do_self_contributions(systems, descriptor)?;

        let keys = descriptor.keys().clone();

        let max_angular = self.parameters.max_angular;
        let max_radial = self.parameters.max_radial;
        let mut contribution = PairContribution::new(max_radial, max_angular, do_gradients.either());

        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.parameters.cutoff)?;
            let species = system.species()?;

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

            for (pair_id, pair) in system.pairs()?.iter().enumerate() {
                let direction = pair.vector / pair.distance;
                self.compute_for_pair(pair.distance, direction, do_gradients, &mut contribution);

                let inverse_cell_pair_vector = Vector3D::new(
                    pair.vector[0] * inverse_cell[0][0] + pair.vector[1] * inverse_cell[1][0] + pair.vector[2] * inverse_cell[2][0],
                    pair.vector[0] * inverse_cell[0][1] + pair.vector[1] * inverse_cell[1][1] + pair.vector[2] * inverse_cell[2][1],
                    pair.vector[0] * inverse_cell[0][2] + pair.vector[1] * inverse_cell[1][2] + pair.vector[2] * inverse_cell[2][2],
                );

                let species_first = species[pair.first];
                let species_second = species[pair.second];
                for spherical_harmonics_l in 0..=self.parameters.max_angular {
                    let block_i = keys.position(&[
                        spherical_harmonics_l.into(),
                        species_first.into(),
                        species_second.into(),
                    ]);

                    if let Some(block_i) = block_i {
                        let sample = &[
                            system_i.into(),
                            pair_id.into(),
                            pair.first.into(),
                            pair.second.into(),
                        ];

                        SphericalExpansionByPair::accumulate_in_block(
                            spherical_harmonics_l,
                            descriptor.block_mut_by_id(block_i),
                            sample,
                            &contribution,
                            do_gradients,
                            inverse_cell_pair_vector,
                        );
                    }
                }

                // also check for the block with a reversed pair, except if
                // we are handling a pair between an atom and it's own
                // periodic image
                if pair.first == pair.second {
                    continue;
                }

                contribution.inverse_pair(&self.m_1_pow_l);

                for spherical_harmonics_l in 0..=self.parameters.max_angular {
                    let block_i = keys.position(&[
                        spherical_harmonics_l.into(),
                        species_second.into(),
                        species_first.into(),
                    ]);

                    if let Some(block_i) = block_i {
                        let sample = &[
                            system_i.into(),
                            pair_id.into(),
                            pair.second.into(),
                            pair.first.into(),
                        ];

                        SphericalExpansionByPair::accumulate_in_block(
                            spherical_harmonics_l,
                            descriptor.block_mut_by_id(block_i),
                            sample,
                            &contribution,
                            do_gradients,
                            -inverse_cell_pair_vector,
                        );
                    }
                }
            }
        }

        Ok(())
    }
}



#[cfg(test)]
mod tests {
    use metatensor::Labels;
    use ndarray::{s, Axis};
    use approx::assert_ulps_eq;

    use crate::systems::test_utils::{test_system, test_systems};
    use crate::Calculator;
    use crate::calculators::{CalculatorBase, SphericalExpansion};

    use super::{SphericalExpansionByPair, SphericalExpansionParameters};
    use super::super::{CutoffFunction, RadialScaling};
    use crate::calculators::radial_basis::RadialBasis;


    fn parameters() -> SphericalExpansionParameters {
        SphericalExpansionParameters {
            cutoff: 3.5,
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
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(SphericalExpansionByPair::new(
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
        let calculator = Calculator::from(Box::new(SphericalExpansionByPair::new(
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
        let calculator = Calculator::from(Box::new(SphericalExpansionByPair::new(
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

        let samples = Labels::new(["structure", "first_atom", "second_atom"], &[
            [0, 1, 2],
            [0, 2, 1],
        ]);

        let keys = Labels::new(["spherical_harmonics_l", "species_atom_1", "species_atom_2"], &[
            [0, -42, 1],
            [0, -42, -42],
            [0, 6, 1], // not part of the default keys
            [0, 1, -42],
            [0, 1, 1],
            [1, -42, -42],
            [1, -42, 1],
            [1, 1, -42],
            [1, 1, 1],
            [2, -42, 1],
            [2, 1, -42],
            [2, 1, 1],
            [2, -42, -42],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn sums_to_spherical_expansion() {
        let mut calculator_by_pair = Calculator::from(Box::new(SphericalExpansionByPair::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);
        let mut calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);
        let expected = calculator.compute(&mut systems, Default::default()).unwrap();

        let by_pair = calculator_by_pair.compute(&mut systems, Default::default()).unwrap();

        // check that keys are the same appart for the names
        assert_eq!(expected.keys().count(), by_pair.keys().count());
        assert_eq!(
            expected.keys().iter().collect::<Vec<_>>(),
            by_pair.keys().iter().collect::<Vec<_>>(),
        );

        for (block, spx) in by_pair.blocks().iter().zip(expected.blocks()) {
            let spx = spx.data();
            let spx_values = spx.values.as_array();

            let block = block.data();
            let values = block.values.as_array();

            for (spx_sample, expected) in spx.samples.iter().zip(spx_values.axis_iter(Axis(0))) {
                let mut sum = ndarray::Array::zeros(expected.raw_dim());

                for (sample_i, &[structure, _, center, _]) in block.samples.iter_fixed_size().enumerate() {
                    if spx_sample[0] == structure && spx_sample[1] == center {
                        sum += &values.slice(s![sample_i, .., ..]);
                    }
                }

                assert_ulps_eq!(sum, expected);
            }
        }
    }
}
