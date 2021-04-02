use ndarray::s;

use approx::assert_relative_eq;

use crate::{CalculationOptions, Calculator, SelectedIndexes};
use crate::system::{System, SimpleSystem};
use crate::system::test_utils::SimpleSystems;
use crate::descriptor::{Descriptor, Indexes, IndexValue};

/// Check that computing a partial subset of features/samples works as intended
/// for the given `calculator` and `systems`.
///
/// This function will check all possible combinations using the given
/// `samples`/`features`. If `gradients` is true, this function also checks the
/// gradients.
#[allow(clippy::needless_pass_by_value)]
pub fn compute_partial(
    mut calculator: Calculator,
    mut systems: SimpleSystems,
    samples: Indexes,
    features: Indexes,
    gradients: bool,
) {
    let mut full = Descriptor::new();
    calculator.compute(&mut systems.get(), &mut full, Default::default()).unwrap();

    // partial set of features, all samples
    let mut partial = Descriptor::new();
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::All,
        selected_features: SelectedIndexes::Some(features.clone()),
        ..Default::default()
    };
    calculator.compute(&mut systems.get(), &mut partial, options).unwrap();

    assert_eq!(full.samples, partial.samples);
    for (partial_i, feature) in features.iter().enumerate() {
        let index = full.features.position(feature).unwrap();
        assert_eq!(
            full.values.slice(s![.., index]),
            partial.values.slice(s![.., partial_i])
        );

        if gradients {
            assert_eq!(
                full.gradients.as_ref().unwrap().slice(s![.., index]),
                partial.gradients.as_ref().unwrap().slice(s![.., partial_i])
            );
        }
    }

    // all features, partial set of samples
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::Some(samples.clone()),
        selected_features: SelectedIndexes::All,
        ..Default::default()
    };
    calculator.compute(&mut systems.get(), &mut partial, options).unwrap();

    assert_eq!(full.features, partial.features);
    for (partial_i, sample) in samples.iter().enumerate() {
        let index = full.samples.position(sample).unwrap();
        assert_eq!(
            full.values.slice(s![index, ..]),
            partial.values.slice(s![partial_i, ..])
        );
    }

    if gradients {
        for (partial_i, sample) in partial.gradients_samples.as_ref().unwrap().iter().enumerate() {
            let index = full.gradients_samples.as_ref().unwrap().position(sample).unwrap();
            assert_eq!(
                full.gradients.as_ref().unwrap().slice(s![index, ..]),
                partial.gradients.as_ref().unwrap().slice(s![partial_i, ..])
            );
        }
    }

    // partial set of features, partial set of samples
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::Some(samples.clone()),
        selected_features: SelectedIndexes::Some(features.clone()),
        ..Default::default()
    };
    calculator.compute(&mut systems.get(), &mut partial, options).unwrap();
    for (sample_i, sample) in samples.iter().enumerate() {
        for (feature_i, feature) in features.iter().enumerate() {
            let full_sample_i = full.samples.position(sample).unwrap();
            let full_feature_i = full.features.position(feature).unwrap();
            assert_eq!(
                full.values[[full_sample_i, full_feature_i]],
                partial.values[[sample_i, feature_i]]
            );
        }
    }

    if gradients {
        for (sample_i, sample) in partial.gradients_samples.as_ref().unwrap().iter().enumerate() {
            for (feature_i, feature) in features.iter().enumerate() {
                let full_sample_i = full.gradients_samples.as_ref().unwrap().position(sample).unwrap();
                let full_feature_i = full.features.position(feature).unwrap();
                assert_eq!(
                    full.gradients.as_ref().unwrap()[[full_sample_i, full_feature_i]],
                    partial.gradients.as_ref().unwrap()[[sample_i, feature_i]]
                );
            }
        }
    }
}

/// Index of an atom moved during finite difference calculation
pub struct MovedAtomIndex {
    /// index of the atomic center in the structure
    pub(crate) center: usize,
    /// spatial dimension index (0 for x, 1 for y and 2 for z)
    pub(crate) spatial: usize,
}

/// Metadata about of one of the gradient sample changed when moving one atom
pub struct ChangedGradientIndex {
    /// Position of the changed sample in the full gradients matrix
    pub(crate) gradient_index: usize,
    /// Sample (NOT gradient) descriptor to which this gradient sample is
    /// associated
    pub(crate) sample: Vec<IndexValue>,
}

/// Check that analytical gradients agree with a finite difference calculation
/// of the gradients.
#[allow(clippy::needless_pass_by_value)]
pub fn finite_difference(
    mut calculator: Calculator,
    mut system: SimpleSystem,
    compute_modified_indexes: impl Fn(&Indexes, MovedAtomIndex) -> Vec<ChangedGradientIndex>,
    max_relative: f64,
) {

    let mut reference = Descriptor::new();
    calculator.compute(&mut [&mut system], &mut reference, Default::default()).unwrap();

    let gradients_samples = reference.gradients_samples.as_ref().unwrap();

    let delta = 1e-9;
    let gradients = reference.gradients.as_ref().unwrap();
    for atom_i in 0..system.size() {
        for spatial in 0..3 {
            system.positions_mut()[atom_i][spatial] += delta;

            let mut updated = Descriptor::new();
            calculator.compute(&mut [&mut system], &mut updated, Default::default()).unwrap();

            let moved = MovedAtomIndex {
                center: atom_i,
                spatial: spatial,
            };

            for changed in compute_modified_indexes(&gradients_samples, moved) {
                let sample_i = reference.samples.position(&changed.sample).expect(
                    "missing sample in reference values"
                );
                assert_eq!(
                    updated.samples.position(&changed.sample).unwrap(),
                    sample_i
                );

                let value = reference.values.slice(s![sample_i, ..]);
                let value_delta = updated.values.slice(s![sample_i, ..]);
                let gradient = gradients.slice(s![changed.gradient_index, ..]);

                assert_eq!(value.shape(), value_delta.shape());
                assert_eq!(value.shape(), gradient.shape());

                let mut finite_difference = value_delta.to_owned().clone();
                finite_difference -= &value;
                finite_difference /= delta;

                assert_relative_eq!(
                    finite_difference, gradient,
                    epsilon=delta,
                    max_relative=max_relative,
                );
            }

            system.positions_mut()[atom_i][spatial] -= delta;
        }
    }
}
