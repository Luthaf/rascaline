use ndarray::Axis;
use approx::{assert_relative_eq, assert_ulps_eq};

use equistore::{Labels, TensorMap};

use crate::calculator::LabelsSelection;
use crate::{CalculationOptions, Calculator};
use crate::systems::{System, SimpleSystem};

/// Check that computing a partial subset of features/samples works as intended
/// for the given `calculator` and `systems`.
///
/// This function will check all possible combinations using the given
/// `samples`/`features`. If `gradients` is true, this function also checks the
/// gradients.
pub fn compute_partial(
    mut calculator: Calculator,
    systems: &mut [Box<dyn System>],
    samples: &Labels,
    properties: &Labels,
) {
    let full = calculator.compute(systems, Default::default()).unwrap();

    check_compute_partial_properties(&mut calculator, &mut *systems, &full, properties);
    check_compute_partial_samples(&mut calculator, &mut *systems, &full, samples);
    check_compute_partial_both(&mut calculator, &mut *systems, &full, samples, properties);
}

fn check_compute_partial_properties(
    calculator: &mut Calculator,
    systems: &mut [Box<dyn System>],
    full: &TensorMap,
    properties: &Labels,
) {
    // partial set of features, all samples
    let options = CalculationOptions {
        selected_samples: LabelsSelection::All,
        selected_properties: LabelsSelection::Subset(properties),
        ..Default::default()
    };
    let partial = calculator.compute(systems, options).unwrap();

    assert_eq!(full.keys(), partial.keys());
    for (full, partial) in full.blocks().iter().zip(partial.blocks()) {
        assert_eq!(full.values().samples, partial.values().samples);
        assert_eq!(full.values().components, partial.values().components);

        assert_eq!(&*partial.values().properties, properties);

        let full_values = full.values().data.as_array();
        let partial_values = partial.values().data.as_array();

        let property_axis = Axis(full_values.shape().len() - 1);

        for (partial_i, property) in properties.iter().enumerate() {
            let property_i = full.values().properties.position(property).unwrap();
            assert_ulps_eq!(
                full_values.index_axis(property_axis, property_i),
                partial_values.index_axis(property_axis, partial_i),
            );

            if let Some(full_gradient) = full.gradients().get("positions") {
                let full_gradient_data = full_gradient.data.as_array();
                let partial_gradient_data = partial.gradients()["positions"].data.as_array();

                let property_axis = Axis(full_gradient_data.shape().len() - 1);
                assert_ulps_eq!(
                    full_gradient_data.index_axis(property_axis, property_i),
                    partial_gradient_data.index_axis(property_axis, partial_i),
                );
            }
        }
    }
}

fn check_compute_partial_samples(
    calculator: &mut Calculator,
    systems: &mut [Box<dyn System>],
    full: &TensorMap,
    samples: &Labels,
) {
    // all features, partial set of samples
    let options = CalculationOptions {
        selected_samples: LabelsSelection::Subset(samples),
        selected_properties: LabelsSelection::All,
        ..Default::default()
    };
    let partial = calculator.compute(systems, options).unwrap();

    assert_eq!(full.keys(), partial.keys());
    for (full, partial) in full.blocks().iter().zip(partial.blocks()) {
        assert_eq!(full.values().components, partial.values().components);
        assert_eq!(full.values().properties, partial.values().properties);

        let full_values = full.values().data.as_array();
        let partial_values = partial.values().data.as_array();

        for (partial_i, sample) in partial.values().samples.iter().enumerate() {
            let sample_i = full.values().samples.position(sample).unwrap();
            assert_ulps_eq!(
                full_values.index_axis(Axis(0), sample_i),
                partial_values.index_axis(Axis(0), partial_i),
            );
        }

        if let Some(full_gradient) = full.gradients().get("positions") {
            let partial_gradient = &partial.gradients()["positions"];

            let full_gradient_data = full_gradient.data.as_array();
            let partial_gradient_data = partial_gradient.data.as_array();

            for (grad_sample_i, grad_sample) in partial_gradient.samples.iter().enumerate() {
                let sample = &partial.values().samples[grad_sample[0].usize()];
                let full_sample_i = full.values().samples.position(sample).unwrap();

                let full_grad_sample_i = full_gradient.samples.position(
                    &[full_sample_i.into(), grad_sample[1], grad_sample[2]]
                ).unwrap();

                assert_ulps_eq!(
                    full_gradient_data.index_axis(Axis(0), full_grad_sample_i),
                    partial_gradient_data.index_axis(Axis(0), grad_sample_i),
                );
            }
        }
    }
}

fn check_compute_partial_both(
    calculator: &mut Calculator,
    systems: &mut [Box<dyn System>],
    full: &TensorMap,
    samples: &Labels,
    properties: &Labels,
) {
    // partial set of features, partial set of samples
    let options = CalculationOptions {
        selected_samples: LabelsSelection::Subset(samples),
        selected_properties: LabelsSelection::Subset(properties),
        ..Default::default()
    };
    let partial = calculator.compute(systems, options).unwrap();

    assert_eq!(full.keys(), partial.keys());
    for (full, partial) in full.blocks().iter().zip(partial.blocks()) {
        assert_eq!(full.values().components, partial.values().components);

        let full_values = full.values().data.as_array();
        let partial_values = partial.values().data.as_array();
        let property_axis = Axis(full_values.shape().len() - 2);

        for (sample_i, sample) in partial.values().samples.iter().enumerate() {
            for (property_i, property) in properties.iter().enumerate() {
                let full_property_i = full.values().properties.position(property).unwrap();
                let full_sample_i = full.values().samples.position(sample).unwrap();

                assert_ulps_eq!(
                    full_values.index_axis(Axis(0), full_sample_i).index_axis(property_axis, full_property_i),
                    partial_values.index_axis(Axis(0), sample_i).index_axis(property_axis, property_i),
                );
            }
        }

        if let Some(full_gradient) = full.gradients().get("positions") {
            let partial_gradient = &partial.gradients()["positions"];

            let full_gradient_data = full_gradient.data.as_array();
            let partial_gradient_data = partial_gradient.data.as_array();

            let property_axis = Axis(full_gradient_data.shape().len() - 2);

            for (grad_sample_i, grad_sample) in partial_gradient.samples.iter().enumerate() {
                let sample = &partial.values().samples[grad_sample[0].usize()];
                let full_sample_i = full.values().samples.position(sample).unwrap();

                let full_grad_sample_i = full_gradient.samples.position(
                    &[full_sample_i.into(), grad_sample[1], grad_sample[2]]
                ).unwrap();

                for (property_i, property) in properties.iter().enumerate() {
                    let full_property_i = full.values().properties.position(property).unwrap();

                    assert_ulps_eq!(
                        full_gradient_data.index_axis(Axis(0), full_grad_sample_i).index_axis(property_axis, full_property_i),
                        partial_gradient_data.index_axis(Axis(0), grad_sample_i).index_axis(property_axis, property_i),
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FinalDifferenceOptions {
    /// distance each atom will be displaced in each direction when computing
    /// finite differences
    pub displacement: f64,
    /// Maximal relative error. 10 * displacement is a good starting point
    pub max_relative: f64,
    /// Threshold below which all values are considered zero. This should be
    /// very small (1e-16) to prevent false positives (if all values & gradients
    /// are below that threshold, tests will pass even with wrong gradients)
    pub epsilon: f64,
}

/// Check that analytical gradients agree with a finite difference calculation
/// of the gradients.
pub fn finite_difference(mut calculator: Calculator, mut system: SimpleSystem, options: FinalDifferenceOptions) {
    let reference = calculator.compute(&mut [Box::new(system.clone())], Default::default()).unwrap();

    for atom_i in 0..system.size().unwrap() {
        for spatial in 0..3 {
            system.positions_mut()[atom_i][spatial] += options.displacement / 2.0;
            let updated_pos = calculator.compute(&mut [Box::new(system.clone())], Default::default()).unwrap();

            system.positions_mut()[atom_i][spatial] -= options.displacement;
            let updated_neg = calculator.compute(&mut [Box::new(system.clone())], Default::default()).unwrap();

            system.positions_mut()[atom_i][spatial] += options.displacement / 2.0;

            assert_eq!(updated_pos.keys(), reference.keys());
            assert_eq!(updated_neg.keys(), reference.keys());

            for (block_i, (_, block)) in reference.iter().enumerate() {
                let gradients = &block.gradients()["positions"];
                let block_pos = &updated_pos.blocks()[block_i];
                let block_neg = &updated_neg.blocks()[block_i];

                for (gradient_i, [sample_i, _, atom]) in gradients.samples.iter_fixed_size().enumerate() {
                    if atom.usize() != atom_i {
                        continue;
                    }
                    let sample_i = sample_i.usize();

                    // check that the same sample is here in both descriptors
                    assert_eq!(block_pos.values().samples[sample_i], block.values().samples[sample_i]);
                    assert_eq!(block_neg.values().samples[sample_i], block.values().samples[sample_i]);

                    let value_pos = block_pos.values().data.as_array().index_axis(Axis(0), sample_i);
                    let value_neg = block_neg.values().data.as_array().index_axis(Axis(0), sample_i);
                    let gradient = gradients.data.as_array().index_axis(Axis(0), gradient_i);
                    let gradient = gradient.index_axis(Axis(0), spatial);

                    assert_eq!(value_pos.shape(), gradient.shape());
                    assert_eq!(value_neg.shape(), gradient.shape());

                    let mut finite_difference = value_pos.to_owned().clone();
                    finite_difference -= &value_neg;
                    finite_difference /= options.displacement;

                    assert_relative_eq!(
                        finite_difference, gradient,
                        epsilon=options.epsilon,
                        max_relative=options.max_relative,
                    );
                }
            }
        }
    }
}
