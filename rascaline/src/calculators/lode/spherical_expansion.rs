use std::collections::BTreeMap;
use std::sync::Arc;

use log::warn;


use equistore::{LabelsBuilder, Labels, LabelValue};
use equistore::TensorMap;

use crate::{Error, System};
use crate::labels::{SamplesBuilder, SpeciesFilter, LongRangePerAtom};
use crate::labels::{KeysBuilder, CenterSingleNeighborsSpeciesKeys};

use super::super::CalculatorBase;
use crate::calculators::soap::RadialBasis;

use crate::math::CachedAllocationsSphericalHarmonics;


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
    spherical_harmonics: CachedAllocationsSphericalHarmonics,
}

impl std::fmt::Debug for LodeSphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl LodeSphericalExpansion {
    pub fn new(parameters: LodeSphericalExpansionParameters) -> Result<LodeSphericalExpansion, Error> {
        let spherical_harmonics = CachedAllocationsSphericalHarmonics::new(
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


        warn!("not implemented yet");
        return Ok(());
    }
}
