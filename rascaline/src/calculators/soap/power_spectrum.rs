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

/// Mapping between the feature index of the power spectrum (indexed with `n1,
/// n2, l`) and the spherical expansion (indexed with `n, l, m`). Each power
/// spectrum feature corresponds to a block (m from -l to l) of spherical
/// expansion features.
struct FeatureBlock {
    /// angular basis number for this feature block. The block size is 2l + 1,
    /// corresponding to all m values from -l to l
    l: isize,
    /// Index of the first feature in the spherical expansion (corresponding to
    /// `n1, l, m=-l`)
    start_n1_l: usize,
    /// Index of the second feature in the spherical expansion (corresponding to
    /// `n2, l, m=-l`)
    start_n2_l: usize,
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

    #[time_graph::instrument(name = "SoapPowerSpectrum::compute")]
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

        // Find out where feature blocks of the spherical expansion are located
        let mut feature_blocks = Vec::with_capacity(descriptor.features.count());
        for feature in descriptor.features.iter() {
            let n1 = feature[0];
            let n2 = feature[1];
            let l = feature[2].isize();

            let start_n1_l = self.spherical_expansion.features.position(
                &[n1, IndexValue::from(l), IndexValue::from(-l)]
            ).expect("missing feature `n1, l, m` in spherical expansion");
            let start_n2_l = self.spherical_expansion.features.position(
                &[n2, IndexValue::from(l), IndexValue::from(-l)]
            ).expect("missing feature `n2, l, m` in spherical expansion");

            feature_blocks.push(FeatureBlock { l, start_n1_l, start_n2_l });
        }

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

            for (feature_i, &FeatureBlock { l, start_n1_l, start_n2_l }) in feature_blocks.iter().enumerate() {
                let mut sum = 0.0;
                for (index_m, m) in (-l..=l).enumerate() {
                    let feature_1 = start_n1_l + index_m;
                    let feature_2 = start_n2_l + index_m;
                    // this code assumes that all m values for a given n/l are
                    // consecutive, let's double check it
                    debug_assert_eq!(self.spherical_expansion.features[feature_1][2].isize(), m);
                    debug_assert_eq!(self.spherical_expansion.features[feature_2][2].isize(), m);

                    unsafe {
                        sum += self.spherical_expansion.values.uget([neighbor_1, feature_1])
                             * self.spherical_expansion.values.uget([neighbor_2, feature_2]);
                    }
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
    use crate::system::test_systems;
    use crate::descriptor::{IndexValue, IndexesBuilder};
    use crate::{Descriptor, Calculator};

    use super::*;
    use super::super::super::CalculatorBase;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::IndexValue::from($value)
        };
    }

    fn parameters(gradients: bool) -> PowerSpectrumParameters {
        PowerSpectrumParameters {
            atomic_gaussian_width: 0.3,
            cutoff: 3.5,
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
            gradients: gradients,
            max_radial: 6,
            max_angular: 6,
            radial_basis: RadialBasis::Gto {}
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters(false)
        )) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor, Default::default()).unwrap();

        assert_eq!(descriptor.environments.names(), ["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        assert_eq!(descriptor.features.names(), ["n1", "n2", "l"]);

        let mut index = 0;
        for n1 in 0..6_usize {
            for n2 in 0..6_usize {
                for l in 0..=6_isize {
                    let expected = [IndexValue::from(n1), IndexValue::from(n2), IndexValue::from(l)];
                    assert_eq!(descriptor.features[index], expected);
                    index += 1;
                }
            }
        }

        // exact values for power spectrum are regression-tested in
        // `rascaline/tests/soap-power-spectrum.rs`
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters(false)
        )) as Box<dyn CalculatorBase>);

        let systems = test_systems(&["water", "methane"]);

        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        samples.add(&[v!(0), v!(1), v!(1), v!(1), v!(1)]);
        samples.add(&[v!(0), v!(2), v!(1), v!(123456), v!(123456)]);
        samples.add(&[v!(1), v!(0), v!(6), v!(1), v!(6)]);
        samples.add(&[v!(1), v!(2), v!(1), v!(1), v!(1)]);
        let samples = samples.finish();

        let mut features = IndexesBuilder::new(vec!["n1", "n2", "l"]);
        features.add(&[v!(0), v!(1), v!(0)]);
        features.add(&[v!(3), v!(3), v!(3)]);
        features.add(&[v!(2), v!(3), v!(2)]);
        features.add(&[v!(1), v!(4), v!(4)]);
        features.add(&[v!(5), v!(2), v!(0)]);
        features.add(&[v!(1), v!(1), v!(2)]);
        let features = features.finish();

        super::super::super::tests_utils::compute_partial(
            calculator, systems, samples, features, false
        );
    }

    // TODO: gradients & finite difference
}
