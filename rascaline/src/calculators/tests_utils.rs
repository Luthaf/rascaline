use ndarray::Axis;
use approx::{assert_relative_eq, assert_ulps_eq};

use metatensor::{Labels, TensorMap, LabelsBuilder};

use crate::calculator::LabelsSelection;
use crate::{CalculationOptions, Calculator};
use crate::systems::{System, SimpleSystem, UnitCell};

/// Check that computing a partial subset of features/samples works as intended
/// for the given `calculator` and `systems`.
///
/// This function will check all possible combinations using the given
/// `samples`/`features`. If `gradients` is true, this function also checks the
/// gradients.
pub fn compute_partial(
    mut calculator: Calculator,
    systems: &mut [Box<dyn System>],
    keys: &Labels,
    samples: &Labels,
    properties: &Labels,
) {
    let full = calculator.compute(systems, Default::default()).unwrap();

    assert_eq!(
        full.keys().intersection(keys, None, None).unwrap().count(),
        full.keys().count(),
        "selected keys should be a superset of the keys, a subset will be created manually"
    );
    check_compute_partial_keys(&mut calculator, &mut *systems, &full, keys);

    assert!(keys.count() > 3, "selected keys should have more than 3 keys");
    let mut subset_keys = LabelsBuilder::new(keys.names());
    for key in keys.iter().take(3) {
        subset_keys.add(key);
    }
    check_compute_partial_keys(&mut calculator, &mut *systems, &full, &subset_keys.finish());

    check_compute_partial_properties(&mut calculator, &mut *systems, &full, properties);
    // check we can remove all properties
    let empty_properties = Labels::empty(properties.names());
    check_compute_partial_properties(&mut calculator, &mut *systems, &full, &empty_properties);

    check_compute_partial_samples(&mut calculator, &mut *systems, &full, samples);
    // check we can remove all samples
    let empty_samples = Labels::empty(samples.names());
    check_compute_partial_samples(&mut calculator, &mut *systems, &full, &empty_samples);

    check_compute_partial_both(&mut calculator, &mut *systems, &full, samples, properties);
}

fn check_compute_partial_keys(
    calculator: &mut Calculator,
    systems: &mut [Box<dyn System>],
    full: &TensorMap,
    keys: &Labels,
) {
    // select keys manually
    let options = CalculationOptions {
        selected_keys: Some(keys),
        ..Default::default()
    };
    let partial = calculator.compute(systems, options).unwrap();

    assert_eq!(partial.keys(), keys);
    for key in keys {
        let mut selected_key = LabelsBuilder::new(keys.names());
        selected_key.add(key);
        let selected_key = selected_key.finish();

        let partial = partial.block(&selected_key).expect("missing block in partial");
        let full = full.block(&selected_key);
        if let Ok(full) = full {
            assert_eq!(full.samples(), partial.samples());
            assert_eq!(full.components(), partial.components());
            assert_eq!(full.properties(), partial.properties());

            let full_values = full.values().to_array();
            let partial_values = partial.values().to_array();
            assert_ulps_eq!(full_values, partial_values);
        }
    }
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
        assert_eq!(full.samples(), partial.samples());
        assert_eq!(full.components(), partial.components());
        assert_eq!(partial.properties(), *properties);

        let full_values = full.values().to_array();
        let partial_values = partial.values().to_array();

        let property_axis = Axis(full_values.shape().len() - 1);

        for (partial_i, property) in properties.iter().enumerate() {
            let property_i = full.properties().position(property).unwrap();
            assert_ulps_eq!(
                full_values.index_axis(property_axis, property_i),
                partial_values.index_axis(property_axis, partial_i),
            );

            if let Some(full_gradient) = full.gradient("positions") {
                let full_gradient_data = full_gradient.values().to_array();
                let partial_gradient_data = partial.gradient("positions").unwrap().values().to_array();

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
        assert_eq!(full.components(), partial.components());
        assert_eq!(full.properties(), partial.properties());

        let full_values = full.values().to_array();
        let partial_values = partial.values().to_array();

        for (partial_i, sample) in partial.samples().iter().enumerate() {
            let sample_i = full.samples().position(sample).unwrap();
            assert_ulps_eq!(
                full_values.index_axis(Axis(0), sample_i),
                partial_values.index_axis(Axis(0), partial_i),
            );
        }

        if let Some(full_gradient) = full.gradient("positions") {
            let partial_gradient = partial.gradient("positions").unwrap();

            let full_gradient_data = full_gradient.values().to_array();
            let partial_gradient_data = partial_gradient.values().to_array();

            for (grad_sample_i, grad_sample) in partial_gradient.samples().iter().enumerate() {
                let sample = &partial.samples()[grad_sample[0].usize()];
                let full_sample_i = full.samples().position(sample).unwrap();

                let full_grad_sample_i = full_gradient.samples().position(
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
        assert_eq!(full.components(), partial.components());

        let full_values = full.values().to_array();
        let partial_values = partial.values().to_array();
        let property_axis = Axis(full_values.shape().len() - 2);

        for (sample_i, sample) in partial.samples().iter().enumerate() {
            for (property_i, property) in properties.iter().enumerate() {
                let full_property_i = full.properties().position(property).unwrap();
                let full_sample_i = full.samples().position(sample).unwrap();

                assert_ulps_eq!(
                    full_values.index_axis(Axis(0), full_sample_i).index_axis(property_axis, full_property_i),
                    partial_values.index_axis(Axis(0), sample_i).index_axis(property_axis, property_i),
                );
            }
        }

        if let Some(full_gradient) = full.gradient("positions") {
            let partial_gradient = partial.gradient("positions").unwrap();

            let full_gradient_data = full_gradient.values().to_array();
            let partial_gradient_data = partial_gradient.values().to_array();

            let property_axis = Axis(full_gradient_data.shape().len() - 2);

            for (grad_sample_i, grad_sample) in partial_gradient.samples().iter().enumerate() {
                let sample = &partial.samples()[grad_sample[0].usize()];
                let full_sample_i = full.samples().position(sample).unwrap();

                let full_grad_sample_i = full_gradient.samples().position(
                    &[full_sample_i.into(), grad_sample[1], grad_sample[2]]
                ).unwrap();

                for (property_i, property) in properties.iter().enumerate() {
                    let full_property_i = full.properties().position(property).unwrap();

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

/// Check that analytical gradients with respect to positions agree with a
/// finite difference calculation of the gradients.
pub fn finite_differences_positions(mut calculator: Calculator, system: &SimpleSystem, options: FinalDifferenceOptions) {
    let calculation_options = CalculationOptions {
        gradients: &["positions"],
        ..Default::default()
    };
    let reference = calculator.compute(&mut [Box::new(system.clone())], calculation_options).unwrap();

    for atom_i in 0..system.size().unwrap() {
        for spatial in 0..3 {
            let mut system_pos = system.clone();
            system_pos.positions_mut()[atom_i][spatial] += options.displacement / 2.0;
            let updated_pos = calculator.compute(&mut [Box::new(system_pos)], Default::default()).unwrap();

            let mut system_neg = system.clone();
            system_neg.positions_mut()[atom_i][spatial] -= options.displacement / 2.0;
            let updated_neg = calculator.compute(&mut [Box::new(system_neg)], Default::default()).unwrap();

            assert_eq!(updated_pos.keys(), reference.keys());
            assert_eq!(updated_neg.keys(), reference.keys());

            for (block_i, (_, block)) in reference.iter().enumerate() {
                let gradients = &block.gradient("positions").unwrap();
                let block_pos = &updated_pos.block_by_id(block_i);
                let block_neg = &updated_neg.block_by_id(block_i);

                for (gradient_i, [sample_i, _, atom]) in gradients.samples().iter_fixed_size().enumerate() {
                    if atom.usize() != atom_i {
                        continue;
                    }
                    let sample_i = sample_i.usize();

                    // check that the same sample is here in both descriptors
                    assert_eq!(block_pos.samples()[sample_i], block.samples()[sample_i]);
                    assert_eq!(block_neg.samples()[sample_i], block.samples()[sample_i]);

                    let value_pos = block_pos.values().to_array().index_axis(Axis(0), sample_i);
                    let value_neg = block_neg.values().to_array().index_axis(Axis(0), sample_i);
                    let gradient = gradients.values().to_array().index_axis(Axis(0), gradient_i);
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


/// Check that analytical gradients with respect to cell agree with a
/// finite difference calculation of the gradients.
pub fn finite_differences_cell(mut calculator: Calculator, system: &SimpleSystem, options: FinalDifferenceOptions) {
    let calculation_options = CalculationOptions {
        gradients: &["cell"],
        ..Default::default()
    };
    let reference = calculator.compute(&mut [Box::new(system.clone())], calculation_options).unwrap();
    let original_cell = system.cell().unwrap().matrix();
    let original_cell_inverse = original_cell.inverse();

    for spatial_1 in 0..3 {
        for spatial_2 in 0..3 {
            let mut deformed_cell = original_cell;
            deformed_cell[spatial_1][spatial_2] += options.displacement / 2.0;

            let mut system_pos = system.clone();
            system_pos.set_cell(UnitCell::from(deformed_cell));
            for position in system_pos.positions_mut() {
                *position = deformed_cell * (original_cell_inverse * *position);
            }
            let updated_pos = calculator.compute(&mut [Box::new(system_pos)], Default::default()).unwrap();

            deformed_cell[spatial_1][spatial_2] -= options.displacement;

            let mut system_neg = system.clone();
            system_neg.set_cell(UnitCell::from(deformed_cell));
            for position in system_neg.positions_mut() {
                *position = deformed_cell * (original_cell_inverse * *position);
            }
            let updated_neg = calculator.compute(&mut [Box::new(system_neg)], Default::default()).unwrap();

            for (block_i, (_, block)) in reference.iter().enumerate() {
                let gradients = &block.gradient("cell").unwrap();
                let block_pos = &updated_pos.block_by_id(block_i);
                let block_neg = &updated_neg.block_by_id(block_i);

                for (gradient_i, [sample_i]) in gradients.samples().iter_fixed_size().enumerate() {
                    let sample_i = sample_i.usize();

                    // check that the same sample is here in both descriptors
                    assert_eq!(block_pos.samples()[sample_i], block.samples()[sample_i]);
                    assert_eq!(block_neg.samples()[sample_i], block.samples()[sample_i]);

                    let value_pos = block_pos.values().to_array().index_axis(Axis(0), sample_i);
                    let value_neg = block_neg.values().to_array().index_axis(Axis(0), sample_i);
                    let gradient = gradients.values().to_array().index_axis(Axis(0), gradient_i);
                    let gradient = gradient.index_axis(Axis(0), spatial_1);
                    let gradient = gradient.index_axis(Axis(0), spatial_2);

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
