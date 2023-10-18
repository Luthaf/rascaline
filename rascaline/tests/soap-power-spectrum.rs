use approx::assert_relative_eq;
use ndarray::{ArrayD, Axis, s};

use metatensor::{Labels, TensorBlockRef};

use rascaline::{Calculator, CalculationOptions};

mod data;

#[test]
fn values() {
    let (mut systems, parameters) = data::load_calculator_input("soap-power-spectrum-values-input.json");

    let mut calculator = Calculator::new("soap_power_spectrum", parameters).unwrap();
    let descriptor = calculator.compute(&mut systems, Default::default()).expect("failed to run calculation");

    let keys_to_move = Labels::empty(vec!["species_center"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = Labels::empty(vec!["species_neighbor_1", "species_neighbor_2"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.block_by_id(0);
    let array = block.values().to_array();

    let mut expected = data::load_expected_values("soap-power-spectrum-values.npy.gz");
    correct_factor(&mut expected, block.properties());

    assert_relative_eq!(array, &expected, max_relative=1e-5);
}

#[test]
fn gradients() {
    let (mut systems, parameters) = data::load_calculator_input("soap-power-spectrum-gradients-input.json");
    let n_atoms = systems.iter().map(|s| s.size().unwrap()).sum();

    let mut calculator = Calculator::new("soap_power_spectrum", parameters).unwrap();
    let options = CalculationOptions {
        gradients: &["positions", "cell"],
        ..Default::default()
    };
    let descriptor = calculator.compute(&mut systems, options).expect("failed to run calculation");

    let keys_to_move = Labels::empty(vec!["species_center"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = Labels::empty(vec!["species_neighbor_1", "species_neighbor_2"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.block_by_id(0);
    let gradients = block.gradient("positions").unwrap();
    let array = sum_gradients(n_atoms, gradients);

    let mut expected = data::load_expected_values("soap-power-spectrum-positions-gradient.npy.gz");
    correct_factor(&mut expected, block.properties());
    assert_relative_eq!(array, &expected, max_relative=1e-6);

    let gradient = block.gradient("cell").unwrap();
    let array = gradient.values().to_array();
    let mut expected = data::load_expected_values("soap-power-spectrum-cell-gradient.npy.gz");
    correct_factor(&mut expected, block.properties());
    assert_relative_eq!(array, &expected, max_relative=1e-6);
}

fn sum_gradients(n_atoms: usize, gradients: TensorBlockRef<'_>) -> ArrayD<f64> {
    assert_eq!(gradients.samples().names(), &["sample", "structure", "atom"]);
    let array = gradients.values().to_array();

    let mut sum = ArrayD::from_elem(vec![n_atoms, 3, gradients.properties().count()], 0.0);
    for ([_, _, atom], row) in gradients.samples().iter_fixed_size().zip(array.axis_iter(Axis(0))) {
        let mut slice = sum.slice_mut(s![atom.usize(), .., ..]);
        slice += &row;
    }

    sum
}


// Add back the missing (-1)^l factor to the reference data
fn correct_factor(reference: &mut ndarray::ArrayD<f64>, properties: Labels) {
    assert!(properties.names() == [ "species_neighbor_1", "species_neighbor_2", "l", "n1", "n2"]);

    let last_axis = reference.shape().len() - 1;

    for (&[_, _, l, _, _], mut column) in properties.iter_fixed_size().zip(reference.axis_iter_mut(Axis(last_axis))) {
        column *= (-1.0_f64).powi(l.i32());
    }
}
