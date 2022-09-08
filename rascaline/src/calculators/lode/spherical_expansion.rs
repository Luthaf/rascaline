use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{Array2, Array3};
use crate::Vector3D;

use equistore::{LabelsBuilder, Labels, LabelValue};
use equistore::TensorMap;

use crate::{Error, System};
use crate::labels::{SamplesBuilder, SpeciesFilter, LongRangePerAtom};
use crate::labels::{KeysBuilder, CenterSingleNeighborsSpeciesKeys};

use crate::math::{KVector, compute_k_vectors};
use crate::systems::UnitCell;

use super::super::CalculatorBase;
use crate::calculators::soap::RadialBasis;

use crate::math::SphericalHarmonicsCache;


/// TODO
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct LodeSphericalExpansionParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// TODO
    pub potential_exponent: usize,
}


/// TODO
pub struct LodeSphericalExpansion {
    /// Parameters governing the spherical expansion
    parameters: LodeSphericalExpansionParameters,
    /// implementation + cached allocation to compute the spherical harmonics
    /// for a single k-vector
    spherical_harmonics: SphericalHarmonicsCache,
}

/// Compute the trigonometric functions for LODE coefficients
struct StructureFactors {
    /// real part of structure factor
    real: Array3<f64>,
    /// imaginary part of structure factor
    imag: Array3<f64>,
}

fn compute_structure_factors(positions: &[Vector3D], k_vectors: &[KVector]) -> StructureFactors {

    let num_atoms: usize = positions.len();
    let num_kvecs: usize = k_vectors.len();

    let mut cosines = Array2::from_elem((num_kvecs, num_atoms), 0.0);
    let mut sines = Array2::from_elem((num_kvecs, num_atoms), 0.0);

    // cosines[i, j] = cos(k_i * r_j), same for sines
    for i_k in 0..num_kvecs {
        for i_p in 0..num_atoms {
            // dot product between kvectors and positions
            let s = k_vectors[i_k].vector[0] * positions[i_p][0] +
                            k_vectors[i_k].vector[1] * positions[i_p][1] +
                            k_vectors[i_k].vector[2] * positions[i_p][2];

            cosines[[i_k, i_p]] = f64::cos(s);
            sines[[i_k, i_p]] = f64::sin(s);
        }
    }

    let mut strucfac = StructureFactors {
        real: Array3::from_elem((num_atoms, num_atoms, num_kvecs), 0.0),
        imag: Array3::from_elem((num_atoms, num_atoms, num_kvecs), 0.0),
    };
    
    for i in 0..num_atoms {
        for j in 0..num_atoms {
            for k in 0..num_kvecs {
                strucfac.real[[i, j, k]] = cosines[[k, i]] * cosines[[k, j]] + sines[[k, i]] * sines[[k, j]];
                strucfac.imag[[i, j, k]] = sines[[k, i]] * cosines[[k, j]] - cosines[[k, i]] * sines[[k, j]];
            }
        }
    }

    strucfac.real *= 2.0;
    strucfac.imag *= 2.0;

    return strucfac
}

impl std::fmt::Debug for LodeSphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl LodeSphericalExpansion {
    pub fn new(parameters: LodeSphericalExpansionParameters) -> Result<LodeSphericalExpansion, Error> {
        let spherical_harmonics = SphericalHarmonicsCache::new(
            parameters.max_angular,
        );

        return Ok(LodeSphericalExpansion {
            parameters,
            spherical_harmonics,
        });
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

        for (system_i, system) in systems.iter_mut().enumerate() {
            let cell = system.cell()?;
            if cell.shape() == UnitCell::infinite().shape() {
                return Err(Error::InvalidParameter("LODE can only be used with periodic systems".into()));
            }
            let k_vectors = compute_k_vectors(&cell, 1.0);

            let strucfac = compute_structure_factors(system.positions()?, &k_vectors);
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_compute_structure_factors() {
        let mut k_vectors = Vec::new();

        k_vectors.push(KVector{vector: Vector3D::new(1.0, 0.0,0.0), norm: 1.0});
        k_vectors.push(KVector{vector: Vector3D::new(0.0, 1.0, 0.0), norm: 1.0});
        
        let positions = [Vector3D::new(1.0, 1.0, 1.0),
                                        Vector3D::new(2.0, 2.0, 2.0)];

        let strucfac = compute_structure_factors(&positions, &k_vectors);

        let ref_real= arr3(
            &[[[2., 2.], [1.0806046117362793, 1.0806046117362793]],
                  [[1.0806046117362793, 1.0806046117362793],[2. , 2. ]]]);
        let ref_imag = arr3(
            &[[[ 0. ,  0. ], [-1.682941969615793, -1.682941969615793]],
                  [[1.682941969615793, 1.682941969615793],[ 0. ,  0. ]]]);

        assert_eq!(strucfac.imag, ref_imag);
        assert_eq!(strucfac.real, ref_real);
    }
}
