use std::collections::BTreeSet;

use ndarray::parallel::prelude::*;

use crate::descriptor::{SamplesBuilder, IndexValue, Indexes, IndexesBuilder};
use crate::descriptor::{TwoBodiesSpeciesSamples, ThreeBodiesSpeciesSamples};

use crate::{CalculationOptions, Calculator, SelectedIndexes};
use crate::{Descriptor, Error, System};

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
pub struct SoapPowerSpectrum {
    parameters: PowerSpectrumParameters,
    spherical_expansion_calculator: Calculator,
    spherical_expansion: Descriptor,
}

impl SoapPowerSpectrum {
    pub fn new(parameters: PowerSpectrumParameters) -> Result<SoapPowerSpectrum, Error> {
        let expansion_parameters = SphericalExpansionParameters {
            cutoff: parameters.cutoff,
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            atomic_gaussian_width: parameters.atomic_gaussian_width,
            gradients: parameters.gradients,
            radial_basis: parameters.radial_basis,
            cutoff_function: parameters.cutoff_function,
        };

        let spherical_expansion = SphericalExpansion::new(expansion_parameters)?;

        return Ok(SoapPowerSpectrum {
            parameters: parameters,
            spherical_expansion_calculator: Calculator::from(
                Box::new(spherical_expansion) as Box<dyn CalculatorBase>
            ),
            spherical_expansion: Descriptor::new(),
        });
    }

    fn get_expansion_samples(&self, samples: &Indexes) -> Indexes {
        assert_eq!(samples.names(), self.samples_builder().names());

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
            TwoBodiesSpeciesSamples::new(1.0).names()
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

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        Box::new(ThreeBodiesSpeciesSamples::with_self_contribution(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        assert_eq!(indexes.names(), self.features_names());
        for value in indexes {
            let n1 = value[0].usize();
            let n2 = value[1].usize();
            let l = value[2].usize();

            if n1 >= self.parameters.max_radial {
                return Err(Error::InvalidParameter(format!(
                    "'n1' is too large for this SoapPowerSpectrum: \
                    expected value below {}, got {}", self.parameters.max_radial, n1
                )))
            }

            if n2 >= self.parameters.max_radial {
                return Err(Error::InvalidParameter(format!(
                    "'n2' is too large for this SoapPowerSpectrum: \
                    expected value below {}, got {}", self.parameters.max_radial, n2
                )))
            }

            if l > self.parameters.max_angular {
                return Err(Error::InvalidParameter(format!(
                    "'l' is too large for this SoapPowerSpectrum: \
                    expected value below {}, got {}", self.parameters.max_angular + 1, l
                )))
            }
        }

        Ok(())
    }

    #[time_graph::instrument(name = "SoapPowerSpectrum::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        assert_eq!(descriptor.samples.names(), self.samples_builder().names());
        assert_eq!(descriptor.features.names(), self.features_names());

        let options = CalculationOptions {
            selected_samples: SelectedIndexes::Some(self.get_expansion_samples(&descriptor.samples)),
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

        let spherical_expansion_samples = &self.spherical_expansion.samples;
        let spherical_expansion_features = &self.spherical_expansion.features;
        let spherical_expansion_values = &self.spherical_expansion.values;

        let samples = &descriptor.samples;

        descriptor.values.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(sample_i, mut value)| {
                let sample = &samples[sample_i];
                let structure = sample[0];
                let center = sample[1];
                let species_center = sample[2];
                let species_neighbor_1 = sample[3];
                let species_neighbor_2 = sample[4];

                let neighbor_1 = spherical_expansion_samples.position(&[
                    structure, center, species_center, species_neighbor_1
                ]).expect("missing data for one of the neighbor species");
                let neighbor_2 = spherical_expansion_samples.position(&[
                    structure, center, species_center, species_neighbor_2
                ]).expect("missing data for one of the neighbor species");

                for (feature_i, block) in feature_blocks.iter().enumerate() {
                    let &FeatureBlock { l, start_n1_l, start_n2_l } = block;

                    let mut sum = 0.0;
                    for (index_m, m) in (-l..=l).enumerate() {
                        let feature_1 = start_n1_l + index_m;
                        let feature_2 = start_n2_l + index_m;
                        // this code assumes that all m values for a given n/l are
                        // consecutive, let's double check it
                        debug_assert_eq!(spherical_expansion_features[feature_1][2].isize(), m);
                        debug_assert_eq!(spherical_expansion_features[feature_2][2].isize(), m);

                        unsafe {
                            sum += spherical_expansion_values.uget([neighbor_1, feature_1])
                                * spherical_expansion_values.uget([neighbor_2, feature_2]);
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
                    value[feature_i] = sum / normalization;
                }
            });

        if self.parameters.gradients {
            let gradients = descriptor.gradients.as_mut().expect("missing power spectrum gradients");
            let gradient_samples = descriptor.gradients_samples.as_ref().expect("missing power spectrum gradient samples");

            let se_gradients_samples = self.spherical_expansion.gradients_samples.as_ref().expect("missing spherical expansion gradient samples");
            let se_gradients = self.spherical_expansion.gradients.as_ref().expect("missing spherical expansion gradients");

            gradients.axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(sample_i, mut gradient)| {
                    let sample = &gradient_samples[sample_i];
                    let structure = sample[0];
                    let center = sample[1];
                    let species_center = sample[2];
                    let species_neighbor_1 = sample[3];
                    let species_neighbor_2 = sample[4];
                    let neighbor = sample[5];
                    let spatial = sample[6];

                    let neighbor_1 = spherical_expansion_samples.position(&[
                        structure, center, species_center, species_neighbor_1
                    ]).expect("missing data for the first neighbor");
                    let neighbor_2 = spherical_expansion_samples.position(&[
                        structure, center, species_center, species_neighbor_2
                    ]).expect("missing data for the second neighbor");

                    let grad_neighbor_1 = se_gradients_samples.position(&[
                        structure, center, species_center, species_neighbor_1, neighbor, spatial
                    ]);
                    let grad_neighbor_2 = se_gradients_samples.position(&[
                        structure, center, species_center, species_neighbor_2, neighbor, spatial
                    ]);

                    for (feature_i, block) in feature_blocks.iter().enumerate() {
                        let &FeatureBlock { l, start_n1_l, start_n2_l } = block;

                        let mut sum = 0.0;
                        for (index_m, m) in (-l..=l).enumerate() {
                            let feature_1 = start_n1_l + index_m;
                            let feature_2 = start_n2_l + index_m;
                            // this code assumes that all m values for a given n/l are
                            // consecutive, let's double check it
                            debug_assert_eq!(spherical_expansion_features[feature_1][2].isize(), m);
                            debug_assert_eq!(spherical_expansion_features[feature_2][2].isize(), m);

                            if let Some(grad_neighbor_1) = grad_neighbor_1 {
                                unsafe {
                                    sum += se_gradients.uget([grad_neighbor_1, feature_1])
                                         * spherical_expansion_values.uget([neighbor_2, feature_2]);
                                }
                            }

                            if let Some(grad_neighbor_2) = grad_neighbor_2 {
                                unsafe {
                                    sum += spherical_expansion_values.uget([neighbor_1, feature_1])
                                         * se_gradients.uget([grad_neighbor_2, feature_2]);
                                }
                            }
                        }

                        if species_neighbor_1 != species_neighbor_2 {
                            // see above
                            sum *= std::f64::consts::SQRT_2;
                        }

                        let normalization = f64::sqrt(2.0 * l as f64 + 1.0);
                        gradient[feature_i] = sum / normalization;
                    }
                });

        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use crate::systems::test_systems;
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
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]).boxed();
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems, &mut descriptor, Default::default()).unwrap();

        assert_eq!(descriptor.samples.names(), ["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
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
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]).boxed();

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
            calculator, &mut systems, samples, features, false
        );
    }

    #[test]
    fn finite_differences() {
        use super::super::super::tests_utils::{MovedAtomIndex, ChangedGradientIndex};

        let calculator = Calculator::from(Box::new(SoapPowerSpectrum::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let system = systems.systems.pop().unwrap();

        let compute_modified_indexes = |gradients_samples: &Indexes, moved: MovedAtomIndex| {
            let mut results = Vec::new();
            for (sample_i, sample) in gradients_samples.iter().enumerate() {
                let neighbor = sample[5];
                let spatial = sample[6];
                if neighbor.usize() == moved.center && spatial.usize() == moved.spatial {
                    results.push(ChangedGradientIndex {
                        gradient_index: sample_i,
                        sample: sample[..5].to_vec(),
                    });
                }
            }
            return results;
        };

        super::super::super::tests_utils::finite_difference(
            calculator, system, compute_modified_indexes
        );
    }
}
