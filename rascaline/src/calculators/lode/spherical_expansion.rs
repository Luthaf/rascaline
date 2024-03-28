use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;
use thread_local::ThreadLocal;
use ndarray::{Array1, Array2, Array3, s};

use metatensor::TensorMap;
use metatensor::{LabelsBuilder, Labels, LabelValue};

use crate::{Error, System, Vector3D};
use crate::systems::UnitCell;

use crate::labels::{SamplesBuilder, AtomicTypeFilter, LongRangeSamplesPerAtom};
use crate::labels::{KeysBuilder, AllTypesPairsKeys};

use super::super::CalculatorBase;

use crate::math::SphericalHarmonicsCache;
use crate::math::{KVector, compute_k_vectors};
use crate::math::{expi, erfc, gamma};

use crate::calculators::radial_basis::RadialBasis;
use super::radial_integral::{LodeRadialIntegralCache, LodeRadialIntegralParameters};

use super::super::{split_tensor_map_by_system, array_mut_for_system};

/// Parameters for LODE spherical expansion calculator.
///
/// The spherical expansion is at the core of representations in the LODE
/// (long-distance equivariant) family. See [this
/// article](https://aip.scitation.org/doi/10.1063/1.5128375) for more
/// information on the LODE representation.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct LodeSphericalExpansionParameters {
    /// Spherical real space cutoff to use for atomic environments.
    /// Note that this cutoff is only used for the projection of the density.
    /// In contrast to SOAP, LODE also takes atoms outside of this cutoff into
    /// account for the density.
    pub cutoff: f64,
    /// Spherical reciprocal cutoff. If `k_cutoff` is `None` a cutoff of `1.2 π
    /// / atomic_gaussian_width`, which is a reasonable value for most systems,
    /// is used.
    pub k_cutoff: Option<f64>,
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density.
    pub atomic_gaussian_width: f64,
    /// Weight of the central atom contribution in the central image to the
    /// features. If `1` the center atom contribution is weighted the same
    /// as any other contribution. If `0` the central atom does not
    /// contribute to the features at all.
    pub center_atom_weight: f64,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Potential exponent of the decorated atom density. Currently only
    /// implemented for potential_exponent < 10. Some exponents can be connected
    /// to SOAP or physics-based quantities: p=0 uses Gaussian densities as in
    /// SOAP, p=1 uses 1/r Coulomb like densities, p=6 uses 1/r^6 dispersion
    /// like densities."
    pub potential_exponent: usize,
}

impl LodeSphericalExpansionParameters {
    /// Get the value of the k-space cutoff (either provided by the user or a
    /// default).
    pub fn get_k_cutoff(&self) -> f64 {
        return self.k_cutoff.unwrap_or(1.2 * std::f64::consts::PI / self.atomic_gaussian_width);
    }
}


/// The actual calculator used to compute LODE spherical expansion coefficients
pub struct LodeSphericalExpansion {
    /// Parameters governing the spherical expansion
    parameters: LodeSphericalExpansionParameters,
    /// implementation + cached allocation to compute spherical harmonics
    spherical_harmonics: ThreadLocal<RefCell<SphericalHarmonicsCache>>,
    /// implementation + cached allocation to compute the radial integral
    radial_integral: ThreadLocal<RefCell<LodeRadialIntegralCache>>,
    /// Cached allocations for the k_vector => nlm projection coefficients.
    /// The vector contains different l values, and the Array is indexed by
    /// `m, n, k`.
    k_vector_to_m_n: ThreadLocal<RefCell<Vec<Array3<f64>>>>,
    /// Cached allocation for everything that only depends on the k vector
    k_dependent_values: ThreadLocal<RefCell<Array1<f64>>>,
}

/// Compute the trigonometric functions for LODE coefficients
struct StructureFactors {
    /// Real part of `e^{i k r}`, the array shape is `(n_atoms, k_vector)`
    real: Array2<f64>,
    /// Imaginary part of `e^{i k r}`, the array shape is `(n_atoms, k_vector)`
    imag: Array2<f64>,
    /// Real part of `\sum_j e^{i k r_i} e^{-i k r_j}`, with one map entry for
    /// each atomic neighbor type. The arrays shape are `(n_atoms, k_vector)`
    real_per_center: BTreeMap<i32, Array2<f64>>,
    /// Imaginary part of `\sum_j e^{i k r_i} e^{-i k r_j}`, with one map entry
    /// for each atomic neighbor type. The arrays shape are `(n_atoms,
    /// k_vector)`
    imag_per_center: BTreeMap<i32, Array2<f64>>,
}

fn compute_structure_factors(positions: &[Vector3D], types: &[i32], k_vectors: &[KVector]) -> StructureFactors {
    let n_atoms = positions.len();
    let n_k_vectors = k_vectors.len();

    let mut cosines = Array2::from_elem((n_atoms, n_k_vectors), 0.0);
    let mut sines = Array2::from_elem((n_atoms, n_k_vectors), 0.0);
    for (atom_i, position) in positions.iter().enumerate() {
        for (ik, k_vector) in k_vectors.iter().enumerate() {
            // dot product between k-vector and positions
            let s = k_vector.norm * k_vector.direction * position;

            cosines[[atom_i, ik]] = f64::cos(s);
            sines[[atom_i, ik]] = f64::sin(s);
        }
    }

    let all_types = types.iter().copied().collect::<BTreeSet<_>>();
    let mut real_per_center = all_types.iter().copied()
        .map(|s| (s, Array2::from_elem((n_atoms, n_k_vectors), 0.0)))
        .collect::<BTreeMap<_, _>>();
    let mut imag_per_center = all_types.iter().copied()
        .map(|s| (s, Array2::from_elem((n_atoms, n_k_vectors), 0.0)))
        .collect::<BTreeMap<_, _>>();

    // Precompute sums of sines and cosines over neighbors (j), depending on the
    // neighbors atomic types
    let mut sum_j_cos = all_types.iter().copied()
        .map(|s| (s, Array1::from_elem(n_k_vectors, 0.0)))
        .collect::<BTreeMap<_, _>>();
    let mut sum_j_sin = all_types.iter().copied()
        .map(|s| (s, Array1::from_elem(n_k_vectors, 0.0)))
        .collect::<BTreeMap<_, _>>();
    for j in 0..n_atoms {
        let sum_j_cos = sum_j_cos.get_mut(&types[j]).unwrap();
        let sum_j_sin = sum_j_sin.get_mut(&types[j]).unwrap();
        for k in 0..n_k_vectors {
            sum_j_cos[k] += cosines[[j, k]];
            sum_j_sin[k] += sines[[j, k]];
        }
    }

    // Compute Sum_j cos(k*r_ij) and Sum_j sin(k*r_ij) using the subtraction theorem
    for i in 0..n_atoms {
        for (neighbor_type, real_per_center) in &mut real_per_center {
            let sum_j_cos = sum_j_cos.get_mut(neighbor_type).unwrap();
            let sum_j_sin = sum_j_sin.get_mut(neighbor_type).unwrap();
            let imag_per_center = imag_per_center.get_mut(neighbor_type).unwrap();
            for k in 0..n_k_vectors {
                let real = cosines[[i, k]] * sum_j_cos[k] + sines[[i, k]] * sum_j_sin[k];
                let imag = sines[[i, k]] * sum_j_cos[k] - cosines[[i, k]] * sum_j_sin[k];
                real_per_center[[i, k]] += 2.0 * real;
                imag_per_center[[i, k]] += 2.0 * imag;
            }
        }
    }

    return StructureFactors {
        real: cosines,
        imag: sines,
        real_per_center: real_per_center,
        imag_per_center: imag_per_center,
    }
}

impl std::fmt::Debug for LodeSphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

/// Resize a 3D ndarray while keeping existing allocation
fn resize_array3(array: &mut ndarray::Array3<f64>, shape: (usize, usize, usize)) {
    let tmp_array = std::mem::take(array);

    let mut data = tmp_array.into_raw_vec();
    data.resize(shape.0 * shape.1 * shape.2, 0.0);
    *array = Array3::from_shape_vec(shape, data).expect("wrong shape");
}

/// Resize a 1D ndarray while keeping existing allocation
fn resize_array1(array: &mut ndarray::Array1<f64>, shape: usize) {
    let tmp_array = std::mem::take(array);

    let mut data = tmp_array.into_raw_vec();
    data.resize(shape, 0.0);
    *array = Array1::from_shape_vec(shape, data).expect("wrong shape");
}

impl LodeSphericalExpansion {
    pub fn new(parameters: LodeSphericalExpansionParameters) -> Result<LodeSphericalExpansion, Error> {
        if parameters.potential_exponent >= 10 {
            return Err(Error::InvalidParameter(
                "LODE is only implemented for potential_exponent < 10".into()
            ));
        }

        // validate the parameters once here, so we are sure we can construct
        // more radial integrals later
        LodeRadialIntegralCache::new(
            parameters.radial_basis.clone(),
            LodeRadialIntegralParameters {
                max_radial: parameters.max_radial,
                max_angular: parameters.max_angular,
                atomic_gaussian_width: parameters.atomic_gaussian_width,
                cutoff: parameters.cutoff,
                k_cutoff: parameters.get_k_cutoff(),
                potential_exponent: parameters.potential_exponent,
            }
        )?;

        return Ok(LodeSphericalExpansion {
            parameters,
            spherical_harmonics: ThreadLocal::new(),
            radial_integral: ThreadLocal::new(),
            k_vector_to_m_n: ThreadLocal::new(),
            k_dependent_values: ThreadLocal::new(),
        });
    }

    fn project_k_to_nlm(&self, k_vectors: &[KVector]) {
        let mut k_vector_to_m_n = self.k_vector_to_m_n.get_or(|| {
            let mut k_vector_to_m_n = Vec::new();
            for _ in 0..=self.parameters.max_angular {
                k_vector_to_m_n.push(Array3::from_elem((0, 0, 0), 0.0));
            }

            return RefCell::new(k_vector_to_m_n);
        }).borrow_mut();

        for o3_lambda in 0..=self.parameters.max_angular {
            let shape = (2 * o3_lambda + 1, self.parameters.max_radial, k_vectors.len());
            resize_array3(&mut k_vector_to_m_n[o3_lambda], shape);
        }

        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = LodeRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                LodeRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                    k_cutoff: self.parameters.get_k_cutoff(),
                    potential_exponent: self.parameters.potential_exponent,
                }
            ).expect("could not create a radial integral");

            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            let spherical_harmonics = SphericalHarmonicsCache::new(self.parameters.max_angular);
            return RefCell::new(spherical_harmonics);
        }).borrow_mut();

        for (ik, k_vector) in k_vectors.iter().enumerate() {
            // we don't need the gradients of spherical harmonics/radial
            // integral w.r.t. k-vectors until we implement gradients w.r.t cell
            radial_integral.compute(k_vector.norm, false);
            spherical_harmonics.compute(k_vector.direction, false);

            for l in 0..=self.parameters.max_angular {
                let spherical_harmonics = spherical_harmonics.values.slice(l as isize);
                let radial_integral = radial_integral.values.slice(s![l, ..]);

                for (m, sph_value) in spherical_harmonics.iter().enumerate() {
                    for (n, ri_value) in radial_integral.iter().enumerate() {
                        k_vector_to_m_n[l][[m, n, ik]] = ri_value * sph_value;
                    }
                }
            }
        }
    }

    #[allow(clippy::float_cmp)]
    fn compute_density_fourier(&self, k_vectors: &[KVector]) -> Array1<f64> {
        let mut fourier = Vec::with_capacity(k_vectors.len());

        let potential_exponent = self.parameters.potential_exponent as f64;
        let smearing_squared = self.parameters.atomic_gaussian_width * self.parameters.atomic_gaussian_width;

        if potential_exponent == 0.0 {
            let factor = (4.0 * std::f64::consts::PI * smearing_squared).powf(0.75);

            for k_vector in k_vectors {
                let value = f64::exp(-0.5 * k_vector.norm * k_vector.norm * smearing_squared);
                fourier.push(factor * value);
            }
        } else if potential_exponent == 1.0 {
            let factor = 4.0 * std::f64::consts::PI;

            for k_vector in k_vectors {
                let k_norm_squared = k_vector.norm * k_vector.norm;
                let value = f64::exp(-0.5 * k_norm_squared * smearing_squared) / k_norm_squared;
                fourier.push(factor * value);
            }
        } else {
            let p_eff = 3.0 - potential_exponent;
            let factor = std::f64::consts::PI.powf(1.5) * (2.0 * smearing_squared).powf(0.5 * p_eff) / gamma(0.5 * potential_exponent);

            for k_vector in k_vectors {
                let k_norm_squared = k_vector.norm * k_vector.norm;
                let x = 0.5 * k_norm_squared * smearing_squared;

                // Compute the gamma_ui over a power law using analytical
                // expressions for a more stable Fourier transform of the
                // density
                let value = if potential_exponent == 2.0 {
                    f64::sqrt(std::f64::consts::PI / x) * erfc(f64::sqrt(x))
                } else if potential_exponent == 3.0 {
                    -expi(-x)
                } else if potential_exponent == 4.0 {
                    2.0 * (f64::exp(-x) - f64::sqrt(std::f64::consts::PI*x) * erfc(f64::sqrt(x)))
                } else if potential_exponent == 5.0 {
                    f64::exp(-x) + x * expi(-x)
                } else if potential_exponent == 6.0 {
                    ((2.0 - 4.0 * x) * f64::exp(-x)
                        + 4.0 * f64::sqrt(std::f64::consts::PI) * x.powf(1.5) * erfc(f64::sqrt(x))) / 3.0
                } else if potential_exponent == 7.0 {
                    (1.0 - x) * f64::exp(-x) / 2.0 - x.powi(2)/2.0 * expi(-x)
                } else if potential_exponent == 8.0 {
                    - 2.0 / 15.0 * ((-3.0 + 2.0 * x - 4.0 * x.powi(2)) * f64::exp(-x)
                        + 4.0 * f64::sqrt(std::f64::consts::PI) * x.powf(2.5) * erfc(f64::sqrt(x)))
                } else if potential_exponent == 9.0 {
                    (x.powi(2) - x + 2.0) * f64::exp(-x) / 6.0 + x.powi(3)/6.0 * expi(-x)
                } else {
                    panic!("potential_exponent = {} is not implemented", potential_exponent);
                };

                fourier.push(factor * value);
            }
        }

        return fourier.into();
    }

    /// Compute k = 0 contributions.
    ///
    /// Values are only non zero for `potential_exponent` = 0 and > 3.
    fn compute_k0_contributions(&self) -> Array1<f64> {
        let atomic_gaussian_width = self.parameters.atomic_gaussian_width;

        let mut k0_contrib = Vec::with_capacity(self.parameters.max_radial);
        let factor = if self.parameters.potential_exponent == 0 {
            let smearing_squared = atomic_gaussian_width * atomic_gaussian_width;

            (2.0 * std::f64::consts::PI * smearing_squared).powf(1.5)
                / (std::f64::consts::PI * smearing_squared).powf(0.75)
                / f64::sqrt(4.0 * std::f64::consts::PI)

        } else if self.parameters.potential_exponent > 3 {
            let potential_exponent = self.parameters.potential_exponent;
            let p_eff = 3. - potential_exponent as f64;

            0.5 * std::f64::consts::PI * 2.0_f64.powf(p_eff)
                / gamma(0.5 * potential_exponent as f64)
                * 2.0_f64.powf((potential_exponent as f64 - 1.0) / 2.0) / -p_eff
                * atomic_gaussian_width.powf(-p_eff)
                / atomic_gaussian_width.powf(2.0 * potential_exponent as f64 - 6.0)
        } else {
            0.0
        };

        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = LodeRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                LodeRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                    k_cutoff: self.parameters.get_k_cutoff(),
                    potential_exponent: self.parameters.potential_exponent,
                }
            ).expect("could not create a radial integral");

            return RefCell::new(radial_integral);
        }).borrow_mut();

        radial_integral.compute(0.0, false);
        for n in 0..self.parameters.max_radial {
            k0_contrib.push(factor * radial_integral.values[[0, n]]);
        }

        return k0_contrib.into();
    }

    /// Compute center atom contribution.
    ///
    /// By symmetry, this only affects the (l, m) = (0, 0) components of the
    /// projection coefficients and only the neighbor type blocks that agrees
    /// with the center atom.
    fn do_center_contribution(&mut self, systems: &mut[Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        let mut radial_integral = self.radial_integral.get_or(|| {
            let radial_integral = LodeRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                LodeRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.cutoff,
                    k_cutoff: self.parameters.get_k_cutoff(),
                    potential_exponent: self.parameters.potential_exponent,
                }
            ).expect("could not create a radial integral");

            return RefCell::new(radial_integral);
        }).borrow_mut();
        radial_integral.compute_center_contribution();

        let central_atom_contrib = &radial_integral.center_contribution;

        for (system_i, system) in systems.iter_mut().enumerate() {
            let types = system.types()?;

            for center_i in 0..system.size()? {
                let block_i = descriptor.keys().position(&[
                    0.into(),
                    1.into(),
                    types[center_i].into(),
                    types[center_i].into(),
                ]);

                if block_i.is_none() {
                    continue;
                }
                let block_i = block_i.expect("we just checked");

                let mut block = descriptor.block_mut_by_id(block_i);
                let block = block.data_mut();
                let array = block.values.to_array_mut();

                let sample = [system_i.into(), center_i.into()];
                let sample_i = match block.samples.position(&sample) {
                    Some(s) => s,
                    None => continue
                };

                for (property_i, [n]) in block.properties.iter_fixed_size().enumerate() {
                    let n = n.usize();
                    array[[sample_i, 0, property_i]] -= (1.0 - self.parameters.center_atom_weight) * central_atom_contrib[n];
                }
            }
        }

        return Ok(());
    }
}

impl CalculatorBase for LodeSphericalExpansion {
    fn name(&self) -> String {
        "lode spherical expansion".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        &[]
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        let builder = AllTypesPairsKeys {};
        let keys = builder.keys(systems)?;

        let mut builder = LabelsBuilder::new(vec!["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        for &[center_type, neighbor_type] in keys.iter_fixed_size() {
            for o3_lambda in 0..=self.parameters.max_angular {
                builder.add(&[o3_lambda.into(), 1.into(), center_type, neighbor_type]);
            }
        }

        return Ok(builder.finish());
    }

    fn sample_names(&self) -> Vec<&str> {
        LongRangeSamplesPerAtom::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);

        // only compute the samples once for each `center_type, neighbor_type`,
        // and re-use the results across `o3_lambda`.
        let mut samples_per_types = BTreeMap::new();
        for [_, _, center_type, neighbor_type] in keys.iter_fixed_size() {
            if samples_per_types.contains_key(&(center_type, neighbor_type)) {
                continue;
            }

            let builder = LongRangeSamplesPerAtom {
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
            "positions" => true,
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        assert_eq!(keys.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([_, _, center_type, neighbor_type], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = LongRangeSamplesPerAtom {
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
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "LodeSphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);

        self.do_center_contribution(systems, descriptor)?;

        let mut descriptors_by_system = split_tensor_map_by_system(descriptor, systems.len());

        // pick either parallelization over systems (if we have a lot of
        // systems) or parallelization over samples.
        //
        // The parallelization over samples has a slightly higher overhead, so
        // if we can avoid it we try to!
        //
        // Trying to use both at the same time would require additional
        // synchronisation to ensure the right threads access per-system
        // pre-computed data (i.e. anything that depends on k-vectors).
        let (parallel_systems, parallel_samples) = if systems.len() > rayon::current_num_threads() {
            (true, false)
        } else {
            (false, true)
        };
        let n_systems = systems.len();

        systems.par_iter_mut()
            .zip_eq(&mut descriptors_by_system)
            .with_min_len(if parallel_systems {1} else {n_systems})
            .enumerate()
            .try_for_each(|(system_i, (system, descriptor))| {
                let types = system.types()?;
                let cell = system.cell()?;
                if cell.shape() == UnitCell::infinite().shape() {
                    return Err(Error::InvalidParameter("LODE can only be used with periodic systems".into()));
                }

                let k_vectors = compute_k_vectors(&cell, self.parameters.get_k_cutoff());
                if k_vectors.is_empty() {
                    return Err(Error::InvalidParameter("No k-vectors for current combination of hyper parameters.".into()));
                }

                self.project_k_to_nlm(&k_vectors);

                let structure_factors = compute_structure_factors(
                    system.positions()?,
                    system.types()?,
                    &k_vectors
                );

                let density_fourier = self.compute_density_fourier(&k_vectors);

                let global_factor = 4.0 * std::f64::consts::PI / cell.volume();

                // Add k = 0 contributions for (m, l) = (0, 0)
                if self.parameters.potential_exponent == 0 || self.parameters.potential_exponent > 3 {
                    let k0_contrib = &self.compute_k0_contributions();
                    for &neighbor_type in types {
                        for center_i in 0..system.size()? {
                            let block_i = descriptor.keys().position(&[
                                0.into(),
                                1.into(),
                                types[center_i].into(),
                                neighbor_type.into(),
                            ]).expect("missing block");

                            let mut block = descriptor.block_mut_by_id(block_i);
                            let data = block.data_mut();
                            let mut array = array_mut_for_system(data.values);

                            let sample = [system_i.into(), center_i.into()];
                            let sample_i = match data.samples.position(&sample) {
                                Some(s) => s,
                                None => continue
                            };

                            for (_property_i, [n]) in data.properties.iter_fixed_size().enumerate() {
                                let n = n.usize();
                                array[[sample_i, 0, _property_i]] += global_factor * k0_contrib[[n]];
                            }
                        }
                    }
                }

                let k_vector_to_m_n = self.k_vector_to_m_n.get()
                    .expect("k_vector_to_m_n should have been created")
                    .borrow();

                // Main loop: Iterate over all blocks, and then all samples in
                // each block (in parallel) to evaluate the projection
                // coefficients
                for (key, mut block) in descriptor.iter_mut() {
                    if block.samples().is_empty() {
                        continue;
                    }

                    let o3_lambda = key[0].usize();
                    let center_type = key[2].i32();
                    let neighbor_type = key[3].i32();

                    let phase = if o3_lambda % 2 == 0 {
                        (-1.0_f64).powi(o3_lambda as i32 / 2)
                    } else {
                        (-1.0_f64).powi((o3_lambda as i32 + 1) / 2)
                    };

                    let sf_per_center_real = &structure_factors.real_per_center[&neighbor_type];
                    let sf_per_center_imag = &structure_factors.imag_per_center[&neighbor_type];

                    let sf_per_center = if o3_lambda % 2 == 0 {
                        sf_per_center_real
                    } else {
                        sf_per_center_imag
                    };

                    let k_vector_to_m_n = &k_vector_to_m_n[o3_lambda];

                    let data = block.data_mut();
                    let samples = &*data.samples;
                    let properties = &*data.properties;

                    array_mut_for_system(data.values)
                        .axis_iter_mut(ndarray::Axis(0))
                        .into_par_iter()
                        .with_min_len(if parallel_samples {1} else {samples.count()})
                        .zip_eq(samples.par_iter())
                        .for_each(|(mut row, sample)| {
                            let center_i = sample[1].usize();

                            if types[center_i] != center_type {
                                // this can happen with sample selection if the
                                // user adds extra samples
                                return;
                            }

                            let sf_center = &sf_per_center.slice(s![center_i, ..]);

                            let mut k_dependent_values = self.k_dependent_values
                                .get_or(|| RefCell::new(Array1::zeros(0)))
                                .borrow_mut();

                            resize_array1(&mut k_dependent_values, k_vectors.len());
                            ndarray::azip!((value in &mut *k_dependent_values, &df in &density_fourier, &sf in sf_center) {
                                *value = global_factor * phase * df * sf;
                            });

                            for m in 0..(2 * o3_lambda + 1) {
                                for (property_i, [n]) in properties.iter_fixed_size().enumerate() {
                                    let n = n.usize();

                                    let mut value = 0.0;
                                    for ik in 0..k_vectors.len() {
                                        // Use unsafe to remove bound checking in release mode with `uget`
                                        // (everything is still bound checked in debug mode).
                                        //
                                        // This divides the calculation time by two for values.
                                        unsafe {
                                            value += k_dependent_values.uget(ik)
                                                   * k_vector_to_m_n.uget([m, n, ik]);
                                        }
                                    }
                                    row[[m, property_i]] += value;
                                }
                            }
                        });

                    if let Some(mut gradient) = block.gradient_mut("positions") {
                        let gradient = gradient.data_mut();
                        let gradient_samples = &*gradient.samples;

                        array_mut_for_system(gradient.values)
                            .axis_iter_mut(ndarray::Axis(0))
                            .into_par_iter()
                            .with_min_len(if parallel_samples {1} else {gradient_samples.count()})
                            .zip_eq(gradient_samples.par_iter())
                            .for_each(|(mut grad_row, grad_sample)| {
                                let sample_i = grad_sample[0].usize();
                                let neighbor_i = grad_sample[2].usize();

                                let sample = &samples[sample_i];
                                let center_i = sample[1].usize();

                                if center_i != neighbor_i {
                                    assert!(types[neighbor_i] == neighbor_type);
                                }
                                assert!(types[center_i] == center_type);

                                let cosines = &structure_factors.real;
                                let sines = &structure_factors.imag;

                                let mut sf_grad_pair = self.k_dependent_values
                                    .get_or(|| RefCell::new(Array1::zeros(0)))
                                    .borrow_mut();

                                resize_array1(&mut sf_grad_pair, k_vectors.len());

                                if o3_lambda % 2 == 0 {
                                    // real part of i*e^{i k (ri - rj)}
                                    for ik in 0..k_vectors.len() {
                                        let factor = sines[[center_i, ik]] * cosines[[neighbor_i, ik]]
                                                   - cosines[[center_i, ik]] * sines[[neighbor_i, ik]];
                                        sf_grad_pair[ik] = 2.0 * factor;
                                    }
                                } else {
                                    // imaginary part of i*e^{i k (ri - rj)}
                                    for ik in 0..k_vectors.len() {
                                        let factor = cosines[[center_i, ik]] * cosines[[neighbor_i, ik]]
                                                   + sines[[center_i, ik]] * sines[[neighbor_i, ik]];
                                        sf_grad_pair[ik] = -2.0 * factor;
                                    }
                                }

                                if center_i == neighbor_i {
                                    // We want the real/imaginary part of `-\sum_{j != i} i*e^{i k (rj - ri)}`
                                    // for which we try to reuse the structure factor per center from
                                    // the values calculation
                                    let (sf_phase, sf_grad) = if o3_lambda % 2 == 0 {
                                        (1.0, sf_per_center_imag)
                                    } else {
                                        (-1.0, sf_per_center_real)
                                    };

                                    let sf_center = &sf_grad.slice(s![center_i, ..]);

                                    // #[allow(clippy::assign_op_pattern)]
                                    if neighbor_type == center_type {
                                        // we need to remove the contribution from the i-i pair
                                        ndarray::azip!((pair in &mut *sf_grad_pair, &center in sf_center) {
                                            *pair = -(sf_phase * center - *pair);
                                        });
                                    } else {
                                        ndarray::azip!((pair in &mut *sf_grad_pair, &center in sf_center) {
                                            *pair = -sf_phase * center;
                                        });
                                    }
                                }

                                // pre-combine everything that only depends on k
                                let mut k_dependent_values = sf_grad_pair;
                                ndarray::azip!((value in &mut *k_dependent_values, &df in &density_fourier) {
                                    *value *= global_factor * phase * df;
                                });

                                for m in 0..(2 * o3_lambda + 1) {
                                    for (property_i, [n]) in properties.iter_fixed_size().enumerate() {
                                        let n = n.usize();

                                        let mut grad = Vector3D::zero();
                                        for (ik, k_vector) in k_vectors.iter().enumerate() {
                                            // Use unsafe to remove bound checking in release mode with `uget`
                                            // (everything is still bound checked in debug mode).
                                            //
                                            // This divides the calculation time by ten for gradients.
                                            unsafe {
                                                grad += k_dependent_values.uget(ik)
                                                      * k_vector_to_m_n.uget([m, n, ik])
                                                      * k_vector.norm
                                                      * k_vector.direction;
                                            }
                                        }

                                        grad_row[[0, m, property_i]] = grad[0];
                                        grad_row[[1, m, property_i]] = grad[1];
                                        grad_row[[2, m, property_i]] = grad[2];
                                    }
                                }
                            });
                    }
                }

                return Ok(());
            }
        )?;

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use crate::Calculator;
    use crate::calculators::CalculatorBase;
    use crate::systems::test_utils::test_system;

    use Vector3D;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn finite_differences_positions() {
        let mut system = test_system("water");
        // FIXME: doing this in the "water" system definition breaks all tests,
        // it should not.
        system.cell = UnitCell::cubic(3.0);

        for p in 0..=6 {
            let calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
                LodeSphericalExpansionParameters {
                    cutoff: 1.0,
                    k_cutoff: None,
                    max_radial: 4,
                    max_angular: 4,
                    atomic_gaussian_width: 1.0,
                    center_atom_weight: 1.0,
                    radial_basis: RadialBasis::splined_gto(1e-8),
                    potential_exponent: p,
                }
            ).unwrap()) as Box<dyn CalculatorBase>);

            let options = crate::calculators::tests_utils::FinalDifferenceOptions {
                displacement: 1e-5,
                max_relative: 1e-4,
                epsilon: 1e-10,
            };
            crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
        }
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
            LodeSphericalExpansionParameters {
                cutoff: 1.0,
                k_cutoff: None,
                max_radial: 4,
                max_angular: 2,
                atomic_gaussian_width: 1.0,
                center_atom_weight: 1.0,
                radial_basis: RadialBasis::splined_gto(1e-8),
                potential_exponent: 1,
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut system = test_system("water");
        system.cell = UnitCell::cubic(3.0);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [2],
        ]);

        let samples = Labels::new(["system", "atom"], &[
            [0, 1],
            [0, 2],
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
            calculator, &mut [Box::new(system)], &keys, &samples, &properties
        );
    }

    #[test]
    fn compute_density_fourier() {
        let k_vectors = [KVector{direction: Vector3D::zero(), norm: 1e-12},
                                       KVector{direction: Vector3D::zero(), norm: 1e-11}];

        // Reference values taken from pyLODE
        let reference_vals = [
            [6.67432573, 6.67432573],  // potential_exponent = 0
            [7.87480497, 7.87480497],  // potential_exponent = 4
            [0.65623375, 0.65623375],  // potential_exponent = 6
        ];

        for (i, &p) in [0, 4, 6].iter().enumerate(){
            let spherical_expansion = LodeSphericalExpansion::new(
                LodeSphericalExpansionParameters {
                    cutoff: 3.5,
                    k_cutoff: None,
                    max_radial: 6,
                    max_angular: 6,
                    atomic_gaussian_width: 1.0,
                    center_atom_weight: 1.0,
                    radial_basis: RadialBasis::splined_gto(1e-8),
                    potential_exponent: p,
                }
            ).unwrap();

            assert_relative_eq!(
                spherical_expansion.compute_density_fourier(&k_vectors),
                arr1(&reference_vals[i]),
                max_relative=1e-8
            );
        }
    }

    #[test]
    fn default_k_cutoff() {
        let atomic_gaussian_width = 0.4;
        let parameters = LodeSphericalExpansionParameters {
            cutoff: 3.5,
            k_cutoff: None,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: atomic_gaussian_width,
            center_atom_weight: 1.0,
            potential_exponent: 1,
            radial_basis: RadialBasis::splined_gto(1e-8),
        };

        assert_eq!(
            parameters.get_k_cutoff(),
            1.2 * std::f64::consts::PI / atomic_gaussian_width
        );
    }

    #[test]
    fn compute_k0_contributions_p0() {
        let spherical_expansion = LodeSphericalExpansion::new(
            LodeSphericalExpansionParameters {
                cutoff: 3.5,
                k_cutoff: None,
                max_radial: 6,
                max_angular: 6,
                atomic_gaussian_width: 0.8,
                center_atom_weight: 1.0,
                potential_exponent: 0,
                radial_basis: RadialBasis::splined_gto(1e-8),
            }
        ).unwrap();

        assert_relative_eq!(
            spherical_expansion.compute_k0_contributions(),
            arr1(&[0.49695, 0.78753, 1.07009, 3.13526, -0.18495, 8.9746]),
            max_relative=1e-4
        );
    }

    #[test]
    fn compute_k0_contributions_p6() {
        let spherical_expansion = LodeSphericalExpansion::new(LodeSphericalExpansionParameters {
            cutoff: 3.5,
            k_cutoff: None,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.8,
            center_atom_weight: 1.0,
            potential_exponent: 6,
            radial_basis: RadialBasis::splined_gto(1e-8),
        }).unwrap();

        assert_relative_eq!(
            spherical_expansion.compute_k0_contributions(),
            arr1(&[0.13337, 0.21136, 0.28719, 0.84143, -0.04964, 2.40858]),
            max_relative=1e-4
        );
    }
}
