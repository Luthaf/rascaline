use std::collections::BTreeSet;

use crate::{CalculationOptions, Calculator, SelectedIndexes, descriptor::{AtomSpeciesEnvironment, EnvironmentIndexes, IndexValue, Indexes, IndexesBuilder}};
use crate::descriptor::ThreeBodiesSpeciesEnvironment;
use crate::{Descriptor, System};

use super::{super::CalculatorBase, SphericalExpansionParameters};
use super::{SphericalExpansion, RadialBasis, CutoffFunction};


/// Parameters for SOAP power spectrum calculator.
///
/// In the SOAP power spectrum, each sample represents rotationally-averaged
/// atomic density correlations, built on top of the spherical expansion. Each
/// sample is a vector indexed by `n1, n2, l`, where `n1` and `n2` are radial
/// basis indexes and `l` is the angular index:
///
/// `< n1 n2 l | X_i > = \sum_m < n1 l m | X_i > < n2 l m | X_i >`
///
/// where the `< n l m | X_i >` are the spherical expansion coefficients.
///
/// See [this review article](https://doi.org/10.1063/1.5090481) for more
/// information on the SOAP representations.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[allow(clippy::module_name_repetitions)]
pub struct PowerSpectrumParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use
    pub max_radial: usize,
    /// Number of spherical harmonics to use
    pub max_angular: usize,
    /// Width of the atom-centered gaussian creating the atomic density
    pub atomic_gaussian_width: f64,
    /// Should we also compute gradients of the feature?
    pub gradients: bool,
    /// radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
}

/// Calculator implementing the Smooth Overlap of Atomic Position (SOAP) power
/// spectrum representation of atomistic systems.
#[allow(clippy::module_name_repetitions)]
pub struct SoapPowerSpectrum {
    parameters: PowerSpectrumParameters,
    spherical_expansion_calculator: Calculator,
    spherical_expansion: Descriptor,
}

impl SoapPowerSpectrum {
    pub fn new(parameters: PowerSpectrumParameters) -> SoapPowerSpectrum {
        let expansion_parameters = SphericalExpansionParameters {
            cutoff: parameters.cutoff,
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            atomic_gaussian_width: parameters.atomic_gaussian_width,
            gradients: parameters.gradients,
            radial_basis: parameters.radial_basis,
            cutoff_function: parameters.cutoff_function,
        };

        let spherical_expansion = SphericalExpansion::new(expansion_parameters);

        return SoapPowerSpectrum {
            parameters: parameters,
            spherical_expansion_calculator: Calculator::from(
                Box::new(spherical_expansion) as Box<dyn CalculatorBase>
            ),
            spherical_expansion: Descriptor::new(),
        };
    }

    fn get_expansion_samples(&self, samples: &Indexes) -> Indexes {
        assert_eq!(samples.names(), self.environments().names());

        let mut set = BTreeSet::new();
        for sample in samples {
            let structure = sample[0];
            let center = sample[1];
            let species_center = sample[2];
            let species_neighbor_1 = sample[3];
            let species_neighbor_2 = sample[4];

            set.insert([
                structure, center, species_center, species_neighbor_1,
            ]);
            set.insert([
                structure, center, species_center, species_neighbor_2,
            ]);
        }

        let mut spherical_expansion_samples = IndexesBuilder::new(
            AtomSpeciesEnvironment::new(1.0).names()
        );
        for index in set {
            spherical_expansion_samples.add(&index);
        }

        return spherical_expansion_samples.finish()
    }

    fn get_expansion_features(&self, features: &Indexes) -> Indexes {
        assert_eq!(features.names(), self.features_names());

        let mut set = BTreeSet::new();
        for feature in features {
            let n1 = feature[0];
            let n2 = feature[1];
            let l = feature[2].isize();

            for m in -l..=l {
                set.insert([n1, IndexValue::from(l), IndexValue::from(m)]);
                set.insert([n2, IndexValue::from(l), IndexValue::from(m)]);
            }
        }

        let mut spherical_expansion_features = IndexesBuilder::new(vec!["n", "l", "m"]);
        for index in set {
            spherical_expansion_features.add(&index);
        }

        return spherical_expansion_features.finish()
    }
}

impl std::fmt::Debug for SoapPowerSpectrum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl CalculatorBase for SoapPowerSpectrum {
    fn name(&self) -> String {
        "SOAP power spectrum".into()
    }

    fn get_parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn features_names(&self) -> Vec<&str> {
        vec!["n1", "n2", "l"]
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(self.features_names());
        for n1 in 0..self.parameters.max_radial {
            for n2 in 0..self.parameters.max_radial {
                for l in 0..(self.parameters.max_angular + 1) {
                    features.add(&[
                        IndexValue::from(n1), IndexValue::from(n2), IndexValue::from(l)
                    ]);
                }
            }
        }
        return features.finish();
    }

    fn environments(&self) -> Box<dyn EnvironmentIndexes> {
        Box::new(ThreeBodiesSpeciesEnvironment::with_self_contribution(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
        assert_eq!(indexes.names(), self.features_names());
        for value in indexes {
            let n1 = value[0].usize();
            let n2 = value[1].usize();
            let l = value[2].usize();
            assert!(n1 < self.parameters.max_radial);
            assert!(n2 < self.parameters.max_radial);
            assert!(l <= self.parameters.max_angular);
        }
    }

    fn check_environments(&self, indexes: &Indexes, systems: &mut [&mut dyn System]) {
        assert_eq!(indexes.names(), self.environments().names());
        // This could be made much faster by not recomputing the full list of
        // potential environments
        let allowed = self.environments().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid environment", value);
        }
    }

    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        assert_eq!(descriptor.environments.names(), self.environments().names());
        assert_eq!(descriptor.features.names(), self.features_names());

        let options = CalculationOptions {
            selected_samples: SelectedIndexes::Some(self.get_expansion_samples(&descriptor.environments)),
            selected_features: SelectedIndexes::Some(self.get_expansion_features(&descriptor.features)),
            ..Default::default()
        };

        self.spherical_expansion_calculator.compute(
            systems,
            &mut self.spherical_expansion,
            options,
        ).expect("failed to compute spherical expansion");

        for (environment_i, environment) in descriptor.environments.iter().enumerate() {
            let structure = environment[0];
            let center = environment[1];
            let species_center = environment[2];
            let species_neighbor_1 = environment[3];
            let species_neighbor_2 = environment[4];

            let neighbor_1 = self.spherical_expansion.environments.position(&[
                structure, center, species_center, species_neighbor_1
            ]).expect("missing data for one of the neighbor species");
            let neighbor_2 = self.spherical_expansion.environments.position(&[
                structure, center, species_center, species_neighbor_2
            ]).expect("missing data for one of the neighbor species");

            for (feature_i, feature) in descriptor.features.iter().enumerate() {
                let n1 = feature[0];
                let n2 = feature[1];
                let l = feature[2].isize();

                let feature_start_n1 = self.spherical_expansion.features.position(
                    &[n1, IndexValue::from(l), IndexValue::from(-l)]
                ).expect("missing feature in spherical expansion");
                let feature_start_n2 = self.spherical_expansion.features.position(
                    &[n2, IndexValue::from(l), IndexValue::from(-l)]
                ).expect("missing feature in spherical expansion");

                let values = &self.spherical_expansion.values;
                let mut sum = 0.0;
                for (index_m, m) in (-l..=l).enumerate() {
                    let feature_1 = feature_start_n1 + index_m;
                    let feature_2 = feature_start_n2 + index_m;
                    // this code assumes that all m values for a given n/l are
                    // consecutive
                    debug_assert_eq!(self.spherical_expansion.features[feature_1][2].isize(), m);
                    debug_assert_eq!(self.spherical_expansion.features[feature_2][2].isize(), m);

                    sum += values[[neighbor_1, feature_1]] * values[[neighbor_2, feature_2]];
                }

                if species_neighbor_1 != species_neighbor_2 {
                    // We only store values for `species_neighbor_1 <
                    // species_neighbor_2` because the values are the same for
                    // pairs `species_neighbor_1 <-> species_neighbor_2` and
                    // `species_neighbor_2 <-> species_neighbor_1`. To ensure
                    // the final kernels are correct, we have to multiply the
                    // corresponding values.
                    sum *= std::f64::consts::SQRT_2;
                }

                let normalization = f64::sqrt(2.0 * l as f64 + 1.0);
                descriptor.values[[environment_i, feature_i]] = sum / normalization;
            }
        }

        if self.parameters.gradients {
            unimplemented!("gradients are not yet implemented");
        }
    }
}


#[cfg(test)]
mod tests {

}
