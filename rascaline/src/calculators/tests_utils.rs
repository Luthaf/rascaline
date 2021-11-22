use std::collections::BTreeSet;

use ndarray::s;

use approx::{assert_relative_eq, assert_ulps_eq};

use crate::{CalculationOptions, Calculator, SelectedIndexes};
use crate::systems::{System, SimpleSystem};
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
    systems: &mut [Box<dyn System>],
    samples: Indexes,
    features: Indexes,
) {
    let mut full = Descriptor::new();
    calculator.compute(systems, &mut full, Default::default()).unwrap();

    // partial set of features, all samples
    let mut partial = Descriptor::new();
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::All,
        selected_features: SelectedIndexes::Subset(features.clone()),
        ..Default::default()
    };
    calculator.compute(systems, &mut partial, options).unwrap();

    assert_eq!(full.samples, partial.samples);
    for (partial_i, feature) in features.iter().enumerate() {
        let index = full.features.position(feature).unwrap();
        assert_ulps_eq!(
            full.values.slice(s![.., index]),
            partial.values.slice(s![.., partial_i])
        );

        if calculator.gradients() {
            assert_ulps_eq!(
                full.gradients.as_ref().unwrap().slice(s![.., index]),
                partial.gradients.as_ref().unwrap().slice(s![.., partial_i])
            );
        }
    }

    // all features, partial set of samples
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::Subset(samples.clone()),
        selected_features: SelectedIndexes::All,
        ..Default::default()
    };
    calculator.compute(systems, &mut partial, options).unwrap();

    assert_eq!(full.features, partial.features);
    for (partial_i, sample) in samples.iter().enumerate() {
        let index = full.samples.position(sample).unwrap();
        assert_ulps_eq!(
            full.values.slice(s![index, ..]),
            partial.values.slice(s![partial_i, ..])
        );
    }

    if calculator.gradients() {
        for (partial_i, sample) in partial.gradients_samples.as_ref().unwrap().iter().enumerate() {
            let index = full.gradients_samples.as_ref().unwrap().position(sample).unwrap();
            assert_ulps_eq!(
                full.gradients.as_ref().unwrap().slice(s![index, ..]),
                partial.gradients.as_ref().unwrap().slice(s![partial_i, ..])
            );
        }
    }

    // partial set of features, partial set of samples
    let options = CalculationOptions {
        selected_samples: SelectedIndexes::Subset(samples.clone()),
        selected_features: SelectedIndexes::Subset(features.clone()),
        ..Default::default()
    };
    calculator.compute(systems, &mut partial, options).unwrap();
    for (sample_i, sample) in samples.iter().enumerate() {
        for (feature_i, feature) in features.iter().enumerate() {
            let full_sample_i = full.samples.position(sample).unwrap();
            let full_feature_i = full.features.position(feature).unwrap();
            assert_ulps_eq!(
                full.values[[full_sample_i, full_feature_i]],
                partial.values[[sample_i, feature_i]]
            );
        }
    }

    if calculator.gradients() {
        for (sample_i, sample) in partial.gradients_samples.as_ref().unwrap().iter().enumerate() {
            for (feature_i, feature) in features.iter().enumerate() {
                let full_sample_i = full.gradients_samples.as_ref().unwrap().position(sample).unwrap();
                let full_feature_i = full.features.position(feature).unwrap();
                assert_ulps_eq!(
                    full.gradients.as_ref().unwrap()[[full_sample_i, full_feature_i]],
                    partial.gradients.as_ref().unwrap()[[sample_i, feature_i]]
                );
            }
        }
    }
}

/// Index of an atom moved during finite difference calculation
#[derive(Copy, Clone, Debug)]
struct MovedAtomIndex {
    /// index of the atomic center in the structure
    center: usize,
    /// spatial dimension index (0 for x, 1 for y and 2 for z)
    spatial: usize,
}

/// Metadata about of one of the gradient sample changed when moving one atom
#[derive(Clone, Debug)]
struct ChangedGradientIndex {
    /// Position of the changed sample in the full gradients matrix
    gradient_index: usize,
    /// Sample (NOT gradient) descriptor to which this gradient sample is
    /// associated
    sample: Vec<IndexValue>,
}

/// Check that analytical gradients agree with a finite difference calculation
/// of the gradients.
#[allow(clippy::needless_pass_by_value)]
pub fn finite_difference(mut calculator: Calculator, mut system: SimpleSystem) {
    let mut reference = Descriptor::new();

    println!("----------- first compute -----------");
    calculator.compute(&mut [Box::new(system.clone())], &mut reference, Default::default()).unwrap();
    println!("----------- first compute end -----------");

    let gradients_samples = reference.gradients_samples.as_ref().unwrap();

    let mut checked_gradient_samples = BTreeSet::new();

    let delta = 1e-6;
    let gradients = reference.gradients.as_ref().unwrap();
    for atom_i in 0..system.size().unwrap() {
        for spatial in 0..3 {
            system.positions_mut()[atom_i][spatial] += delta / 2.0;
            let mut updated_pos = Descriptor::new();
            calculator.compute(&mut [Box::new(system.clone())], &mut updated_pos, Default::default()).unwrap();

            system.positions_mut()[atom_i][spatial] -= delta;
            let mut updated_neg = Descriptor::new();
            calculator.compute(&mut [Box::new(system.clone())], &mut updated_neg, Default::default()).unwrap();

            let moved = MovedAtomIndex {
                center: atom_i,
                spatial: spatial,
            };

            for changed in compute_modified_indexes(gradients_samples, moved) {
                let sample_i = reference.samples.position(&changed.sample).expect(
                    "missing sample in reference values"
                );
                assert_eq!(
                    updated_pos.samples.position(&changed.sample).unwrap(),
                    sample_i
                );
                assert_eq!(
                    updated_neg.samples.position(&changed.sample).unwrap(),
                    sample_i
                );

                checked_gradient_samples.insert(changed.gradient_index);

                let value_pos = updated_pos.values.slice(s![sample_i, ..]);
                let value_neg = updated_neg.values.slice(s![sample_i, ..]);
                let gradient = gradients.slice(s![changed.gradient_index, ..]);

                assert_eq!(value_pos.shape(), value_neg.shape());
                assert_eq!(value_pos.shape(), gradient.shape());

                let mut finite_difference = value_pos.to_owned().clone();
                finite_difference -= &value_neg;
                finite_difference /= delta;

                assert_relative_eq!(
                    finite_difference, gradient,
                    epsilon=delta,
                    max_relative=delta,
                );
            }

            system.positions_mut()[atom_i][spatial] += delta / 2.0;
        }
    }

    // ensure that all values in the gradient have been checked
    assert_eq!(checked_gradient_samples.len(), gradients.shape()[0]);
}

fn compute_modified_indexes(gradients_samples: &Indexes, moved: MovedAtomIndex) -> Vec<ChangedGradientIndex> {
    let sample_size = gradients_samples.size() - 2;

    assert!(["neighbor", "atom"].contains(&gradients_samples.names()[sample_size]));
    assert!(gradients_samples.names()[sample_size + 1] == "spatial");

    let mut results = Vec::new();
    for (sample_i, sample) in gradients_samples.iter().enumerate() {
        let neighbor = sample[sample_size];
        let spatial = sample[sample_size + 1];
        if neighbor.usize() == moved.center && spatial.usize() == moved.spatial {
            results.push(ChangedGradientIndex {
                gradient_index: sample_i,
                sample: sample[..sample_size].to_vec(),
            });
        }
    }
    return results;
}
