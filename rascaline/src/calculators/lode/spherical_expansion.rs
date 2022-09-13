use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, s};

use equistore::TensorMap;
use equistore::{LabelsBuilder, Labels, LabelValue};

use crate::{Error, System, Vector3D};
use crate::systems::UnitCell;

use crate::labels::{SamplesBuilder, SpeciesFilter, LongRangePerAtom};
use crate::labels::{KeysBuilder, CenterSingleNeighborsSpeciesKeys};

use super::super::CalculatorBase;

use crate::math::SphericalHarmonicsCache;
use crate::math::{KVector, compute_k_vectors};
use crate::math::{gamma_ui, gamma};

use crate::calculators::radial_basis::RadialBasis;
use super::radial_integral::{LodeRadialIntegralCache, LodeRadialIntegralParameters};

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
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Potential exponent of the decorated atom density. Currently only
    /// implemented for potential_exponent <= 6. Some exponents can be connected
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
    spherical_harmonics: SphericalHarmonicsCache,
    /// implementation + cached allocation to compute the radial integral
    radial_integral: LodeRadialIntegralCache,
    /// Cached allocations for the k_vector => nlm projection coefficients.
    /// The vector contains different l values, and the Array is indexed by
    /// `m, n, k`.
    k_vector_to_m_n: Vec<Array3<f64>>,
}

/// Compute the trigonometric functions for LODE coefficients
struct StructureFactors {
    /// Real part of `e^{i k r}`, the array shape is `(n_atoms, k_vector)`
    real: Array2<f64>,
    /// Imaginary part of `e^{i k r}`, the array shape is `(n_atoms, k_vector)`
    imag: Array2<f64>,
    /// Real part of `\sum_j e^{i k r_i} e^{-i k r_j}`, with one map entry for
    /// each species of the atom j. The arrays shape are `(n_atoms, k_vector)`
    real_per_center: BTreeMap<i32, Array2<f64>>,
    /// Imaginary part of `\sum_j e^{i k r_i} e^{-i k r_j}`, with one map entry
    /// for each species of the atom j. The arrays shape are `(n_atoms,
    /// k_vector)`
    imag_per_center: BTreeMap<i32, Array2<f64>>,
}

fn compute_structure_factors(positions: &[Vector3D], species: &[i32], k_vectors: &[KVector]) -> StructureFactors {
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

    let all_species = species.iter().copied().collect::<BTreeSet<_>>();
    let mut real_per_center = all_species.iter().copied()
        .map(|s| (s, Array2::from_elem((n_atoms, n_k_vectors), 0.0)))
        .collect::<BTreeMap<_, _>>();
    let mut imag_per_center = all_species.iter().copied()
        .map(|s| (s, Array2::from_elem((n_atoms, n_k_vectors), 0.0)))
        .collect::<BTreeMap<_, _>>();

    for i in 0..n_atoms {
        for j in 0..n_atoms {
            for k in 0..n_k_vectors {
                let real = cosines[[i, k]] * cosines[[j, k]] + sines[[i, k]] * sines[[j, k]];
                let imag = sines[[i, k]] * cosines[[j, k]] - cosines[[i, k]] * sines[[j, k]];

                let real_per_center = real_per_center.get_mut(&species[j]).unwrap();
                let imag_per_center = imag_per_center.get_mut(&species[j]).unwrap();
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

impl LodeSphericalExpansion {
    pub fn new(parameters: LodeSphericalExpansionParameters) -> Result<LodeSphericalExpansion, Error> {
        if parameters.potential_exponent > 6 {
            return Err(Error::InvalidParameter(
                "LODE is only implemented for potential_exponent <= 6".into()
            ));
        }

        let spherical_harmonics = SphericalHarmonicsCache::new(
            parameters.max_angular,
        );

        let radial_integral = LodeRadialIntegralCache::new(
            parameters.radial_basis,
            LodeRadialIntegralParameters {
                max_radial: parameters.max_radial,
                max_angular: parameters.max_angular,
                atomic_gaussian_width: parameters.atomic_gaussian_width,
                cutoff: parameters.cutoff,
                k_cutoff: parameters.get_k_cutoff(),
            })?;

        let mut k_vector_to_m_n = Vec::new();
        for _ in 0..=parameters.max_angular {
            k_vector_to_m_n.push(Array3::from_elem((0, 0, 0), 0.0));
        }

        return Ok(LodeSphericalExpansion {
            parameters,
            spherical_harmonics,
            radial_integral,
            k_vector_to_m_n,
        });
    }

    fn project_k_to_nlm(&mut self, k_vectors: &[KVector]) {
        for spherical_harmonics_l in 0..=self.parameters.max_angular {
            let shape = (2 * spherical_harmonics_l + 1, self.parameters.max_radial, k_vectors.len());

            // resize the arrays while keeping existing allocations
            let array = std::mem::take(&mut self.k_vector_to_m_n[spherical_harmonics_l]);

            let mut data = array.into_raw_vec();
            data.resize(shape.0 * shape.1 * shape.2, 0.0);
            let array = Array3::from_shape_vec(shape, data).expect("wrong shape");

            self.k_vector_to_m_n[spherical_harmonics_l] = array;
        }


        for (ik, k_vector) in k_vectors.iter().enumerate() {
            self.radial_integral.compute(k_vector.norm, false);
            // we don't need the gradients of spherical harmonics w.r.t. k-vectors
            self.spherical_harmonics.compute(k_vector.direction, false);

            for l in 0..=self.parameters.max_angular {
                let spherical_harmonics = self.spherical_harmonics.values.slice(l as isize);
                let radial_integral = self.radial_integral.values.slice(s![l, ..]);

                for (m, sph_value) in spherical_harmonics.iter().enumerate() {
                    for (n, ri_value) in radial_integral.iter().enumerate() {
                        self.k_vector_to_m_n[l][[m, n, ik]] = ri_value * sph_value;
                    }
                }
            }
        }
    }

    fn compute_density_fourrier(&self, k_vectors: &[KVector]) -> Array1<f64> {
        let mut fourrier = Vec::new();
        fourrier.reserve(k_vectors.len());

        let potential_exponent = self.parameters.potential_exponent;
        let smearing_squared = self.parameters.atomic_gaussian_width * self.parameters.atomic_gaussian_width;

        if potential_exponent == 0 {
            let factor = (4.0 * std::f64::consts::PI * smearing_squared).powf(0.75);

            for k_vector in k_vectors {
                let value = f64::exp(-0.5 * k_vector.norm * k_vector.norm * smearing_squared);
                fourrier.push(factor * value);
            }
        } else if potential_exponent == 1 {
            let factor = 4.0 * std::f64::consts::PI;

            for k_vector in k_vectors {
                let k_norm_squared = k_vector.norm * k_vector.norm;
                let value = f64::exp(-0.5 * k_norm_squared * smearing_squared) / k_norm_squared;
                fourrier.push(factor * value);
            }
        } else {
            let p_eff = 3 - potential_exponent;
            let factor = std::f64::consts::PI.powf(1.5) * 2.0_f64.powi(p_eff as i32) / gamma(0.5 * potential_exponent as f64);

            for k_vector in k_vectors {
                let k_norm_squared = k_vector.norm * k_vector.norm;
                let value = gamma_ui(0.5 * p_eff as f64, 0.5 * k_norm_squared * smearing_squared);
                fourrier.push(factor * value / k_vector.norm.powi(p_eff as i32));
            }
        }
        return fourrier.into();
    }

    /// Compute k = 0 contributions.
    ///
    /// Values are only non zero for `potential_exponent` = 0 and >= 4.
    fn compute_k0_contributions(&mut self) -> Array1<f64> {
        let atomic_gaussian_width = self.parameters.atomic_gaussian_width;

        let mut k0_contrib = Vec::new();
        k0_contrib.reserve(self.parameters.max_radial);
        let factor = if self.parameters.potential_exponent == 0 {
            let smearing_squared = atomic_gaussian_width * atomic_gaussian_width;

            (2.0 * std::f64::consts::PI * smearing_squared).powf(1.5)
                / (std::f64::consts::PI * smearing_squared).powf(0.75)
                / f64::sqrt(4.0 * std::f64::consts::PI)

        } else if self.parameters.potential_exponent >= 4 {
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

        self.radial_integral.compute(0.0, false);
        for n in 0..self.parameters.max_radial {
            k0_contrib.push(factor * self.radial_integral.values[[0, n]]);
        }

        return k0_contrib.into();
    }
}

impl CalculatorBase for LodeSphericalExpansion {
    fn name(&self) -> String {
        "lode spherical expansion".into()
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
        LongRangePerAtom::samples_names()
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

            let builder = LongRangePerAtom {
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
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Arc<Labels>], systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        assert_eq!(keys.names(), ["spherical_harmonics_l", "species_center", "species_neighbor"]);
        assert_eq!(keys.count(), samples.len());

        let mut gradient_samples = Vec::new();
        for ([_, species_center, species_neighbor], samples) in keys.iter_fixed_size().zip(samples) {
            let builder = LongRangePerAtom {
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

    #[time_graph::instrument(name = "LodeSphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {

        let k_cutoff = self.parameters.get_k_cutoff();

        for (system_i, system) in systems.iter_mut().enumerate() {
            let species = system.species()?;
            let cell = system.cell()?;
            if cell.shape() == UnitCell::infinite().shape() {
                return Err(Error::InvalidParameter("LODE can only be used with periodic systems".into()));
            }

            let k_vectors = compute_k_vectors(&cell, k_cutoff);
            if k_vectors.is_empty() {
                return Err(Error::InvalidParameter("No k-vectors for current combination of hyper parameters.".into()));
            }

            assert!(!k_vectors.is_empty());

            self.project_k_to_nlm(&k_vectors);

            let structure_factors = compute_structure_factors(
                system.positions()?,
                system.species()?,
                &k_vectors
            );

            let density_fourrier = self.compute_density_fourrier(&k_vectors);

            let global_factor = 4.0 * std::f64::consts::PI / cell.volume();

            // Add k = 0 contributions for (m, l) = (0, 0)
            if self.parameters.potential_exponent == 0 || self.parameters.potential_exponent >= 4 {
                let k0_contrib = &self.compute_k0_contributions();
                for &species_neighbor in species {
                    for center_i in 0..system.size()? {
                        let block_i = descriptor.keys().position(&[
                            0.into(),
                            species[center_i].into(),
                            species_neighbor.into(),
                        ]).expect("missing block");

                        let mut block = descriptor.block_mut_by_id(block_i);
                        let values = block.values_mut();
                        let array = values.data.as_array_mut();

                        let sample = [system_i.into(), center_i.into()];
                        let sample_i = match values.samples.position(&sample) {
                            Some(s) => s,
                            None => continue
                        };

                        for (_property_i, [n]) in values.properties.iter_fixed_size().enumerate() {
                            let n = n.usize();
                            array[[sample_i, 0, _property_i]] += global_factor * k0_contrib[[n]];
                        }
                    }
                }
            }

            for spherical_harmonics_l in 0..=self.parameters.max_angular {
                let phase = if spherical_harmonics_l % 2 == 0 {
                    (-1.0_f64).powi(spherical_harmonics_l as i32 / 2)
                } else {
                    (-1.0_f64).powi((spherical_harmonics_l as i32 + 1) / 2)
                };

                let sf_per_center = if spherical_harmonics_l % 2 == 0 {
                    &structure_factors.real_per_center
                } else {
                    &structure_factors.imag_per_center
                };

                let k_vector_to_m_n = &self.k_vector_to_m_n[spherical_harmonics_l];

                for (&species_neighbor, sf_per_center) in sf_per_center.iter() {
                    for center_i in 0..system.size()? {
                        let block_i = descriptor.keys().position(&[
                            spherical_harmonics_l.into(),
                            species[center_i].into(),
                            species_neighbor.into(),
                        ]).expect("missing block");
                        let mut block = descriptor.block_mut_by_id(block_i);
                        let values = block.values_mut();
                        let array = values.data.as_array_mut();

                        let sample = [system_i.into(), center_i.into()];
                        let sample_i = match values.samples.position(&sample) {
                            Some(s) => s,
                            None => continue
                        };

                        for m in 0..(2 * spherical_harmonics_l + 1) {
                            for (property_i, [n]) in values.properties.iter_fixed_size().enumerate() {
                                let n = n.usize();

                                let mut value = 0.0;
                                for ik in 0..k_vectors.len() {
                                    // Use unsafe to remove bound checking in
                                    // release mode with `uget` (everything is
                                    // still bound checked in debug mode).
                                    //
                                    // This divides the calculation time by two
                                    // for values.
                                    unsafe {
                                        value += global_factor * phase
                                            * density_fourrier.uget(ik)
                                            * sf_per_center.uget([center_i, ik])
                                            * k_vector_to_m_n.uget([m, n, ik]);
                                    }
                                }
                                array[[sample_i, m, property_i]] += value;
                            }
                        }

                        if let Some(ref mut gradients) = block.gradient_mut("positions") {
                            for (neighbor_i, &current_neighbor_species) in species.iter().enumerate() {
                                if neighbor_i == center_i {
                                    continue;
                                }

                                if current_neighbor_species != species_neighbor {
                                    continue;
                                }

                                let array = gradients.data.as_array_mut();

                                let grad_sample_self_i = gradients.samples.position(&[
                                    sample_i.into(), system_i.into(), center_i.into()
                                ]).expect("missing self gradient sample");

                                let grad_sample_other_i = gradients.samples.position(&[
                                    sample_i.into(), system_i.into(), neighbor_i.into()
                                ]).expect("missing gradient sample");

                                let mut sf_grad = Vec::with_capacity(k_vectors.len());
                                let cosines = &structure_factors.real;
                                let sines = &structure_factors.imag;
                                if spherical_harmonics_l % 2 == 0 {
                                    let i = center_i;
                                    let j = neighbor_i;
                                    // real part of i*e^{i k (rj - ri)}
                                    for ik in 0..k_vectors.len() {
                                        let factor = sines[[i, ik]] * cosines[[j, ik]]
                                            - cosines[[i, ik]] * sines[[j, ik]];
                                        sf_grad.push(2.0 * factor);
                                    }
                                } else {
                                    let i = center_i;
                                    let j = neighbor_i;
                                    // imaginary part of i*e^{i k (rj - ri)}
                                    for ik in 0..k_vectors.len() {
                                        let factor = cosines[[i, ik]] * cosines[[j, ik]]
                                            + sines[[i, ik]] * sines[[j, ik]];
                                        sf_grad.push(-2.0 * factor);
                                    }
                                }
                                let sf_grad = Array1::from(sf_grad);

                                for m in 0..(2 * spherical_harmonics_l + 1) {
                                    for (property_i, [n]) in gradients.properties.iter_fixed_size().enumerate() {
                                        let n = n.usize();

                                        let mut grad = Vector3D::zero();
                                        for (ik, k_vector) in k_vectors.iter().enumerate() {
                                            // Use unsafe to remove bound
                                            // checking in release mode with
                                            // `uget` (everything is still bound
                                            // checked in debug mode).
                                            //
                                            // This divides the calculation time
                                            // by ten for gradients.
                                            unsafe {
                                                grad += global_factor * phase
                                                    * density_fourrier.uget(ik)
                                                    * sf_grad.uget(ik)
                                                    * k_vector_to_m_n.uget([m, n, ik])
                                                    * k_vector.norm
                                                    * k_vector.direction;
                                            }
                                        }

                                        array[[grad_sample_other_i, 0, m, property_i]] += grad[0];
                                        array[[grad_sample_other_i, 1, m, property_i]] += grad[1];
                                        array[[grad_sample_other_i, 2, m, property_i]] += grad[2];

                                        array[[grad_sample_self_i, 0, m, property_i]] -= grad[0];
                                        array[[grad_sample_self_i, 1, m, property_i]] -= grad[1];
                                        array[[grad_sample_self_i, 2, m, property_i]] -= grad[2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use crate::{Calculator, CalculationOptions};
    use crate::calculators::{CalculatorBase, SphericalExpansion, SphericalExpansionParameters};
    use crate::calculators::soap::{CutoffFunction, RadialScaling};
    use crate::systems::test_utils::{test_system, test_systems};

    use approx::assert_relative_eq;
    use ndarray::arr1;

    use super::*;

    #[test]
    fn finite_differences_positions() {
        let calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
            LodeSphericalExpansionParameters {
                cutoff: 3.5,
                k_cutoff: None,
                max_radial: 6,
                max_angular: 6,
                atomic_gaussian_width: 0.8,
                potential_exponent: 1,
                radial_basis: RadialBasis::splined_gto(1e-8),
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut system = test_system("water");
        // FIXME: doing this in the "water" system definition breaks all tests,
        // it should not.
        system.cell = UnitCell::cubic(3.0);


        let options = crate::calculators::tests_utils::FinalDifferenceOptions {
            displacement: 1e-5,
            max_relative: 1e-4,
            epsilon: 1e-10,
        };
        crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
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
        let mut spherical_expansion = LodeSphericalExpansion::new(
            LodeSphericalExpansionParameters {
                cutoff: 3.5,
                k_cutoff: None,
                max_radial: 6,
                max_angular: 6,
                atomic_gaussian_width: 0.8,
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
        let mut spherical_expansion = LodeSphericalExpansion::new(LodeSphericalExpansionParameters {
            cutoff: 3.5,
            k_cutoff: None,
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.8,
            potential_exponent: 6,
            radial_basis: RadialBasis::splined_gto(1e-8),
        }).unwrap();

        assert_relative_eq!(
            spherical_expansion.compute_k0_contributions(),
            arr1(&[0.13337, 0.21136, 0.28719, 0.84143, -0.04964, 2.40858]),
            max_relative=1e-4
        );
    }

    #[test]
    fn soap_lode() {
        // TODO: this test is a bit slow (more than 10s) in debug mode, should
        // we move it to regression tests so that it runs in release mode?
        let mut systems = test_systems(&["tetramer"]);

        let lode_parameters = LodeSphericalExpansionParameters {
            cutoff: 6.0,
            k_cutoff: Some(12.0),
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            potential_exponent: 0,
            radial_basis: RadialBasis::splined_gto(1e-8),
        };

        let soap_parameters = SphericalExpansionParameters {
            cutoff: lode_parameters.cutoff,
            max_radial: lode_parameters.max_radial,
            max_angular: lode_parameters.max_angular,
            atomic_gaussian_width: lode_parameters.atomic_gaussian_width,
            center_atom_weight: 1.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
            cutoff_function: CutoffFunction::Step {},
            radial_scaling: RadialScaling::None {},
        };

        let mut lode_calculator = Calculator::from(Box::new(LodeSphericalExpansion::new(
            lode_parameters
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut soap_calculator = Calculator::from(Box::new(SphericalExpansion::new(
            soap_parameters
        ).unwrap()) as Box<dyn CalculatorBase>);

        let options = CalculationOptions {..Default::default()};

        let mut lode_descriptor = lode_calculator.compute(&mut systems, options).unwrap();
        let mut soap_descriptor = soap_calculator.compute(&mut systems, options).unwrap();

        let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
        lode_descriptor.keys_to_samples(&keys_to_move, true).unwrap();
        soap_descriptor.keys_to_samples(&keys_to_move, true).unwrap();

        let keys_to_move = LabelsBuilder::new(vec!["species_neighbor"]).finish();
        lode_descriptor.keys_to_properties(&keys_to_move, true).unwrap();
        soap_descriptor.keys_to_properties(&keys_to_move, true).unwrap();

        // TODO: check all blocks
        assert_relative_eq!(
            lode_descriptor.blocks()[0].values().data.as_array(),
            soap_descriptor.blocks()[0].values().data.as_array(),
            // TODO: we can decrease this by using more k-vectors, but this
            // makes the test run even slower
            max_relative=1e-2
        );
    }
}
