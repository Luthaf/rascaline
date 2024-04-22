use approx::assert_relative_eq;
use ndarray::{ArrayD, Axis, s};

use metatensor::{Labels, TensorBlockRef};

use rascaline::{CalculationOptions, Calculator};

mod data;

#[test]
fn values_no_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-values-input.json");

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    let descriptor = calculator.compute(&mut systems, Default::default()).expect("failed to run calculation");

    let keys_to_move = Labels::empty(vec!["center_type"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = Labels::empty(vec!["neighbor_type"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    let descriptor = descriptor.components_to_properties(&["o3_mu"]).unwrap();

    let keys_to_move = Labels::empty(vec!["o3_lambda"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.block_by_id(0);
    let array = block.values().to_array();

    let expected = &data::load_expected_values("spherical-expansion-values.npy.gz");
    assert_relative_eq!(array, expected, max_relative=5e-5);
}

#[test]
fn values_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-pbc-values-input.json");

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    let descriptor = calculator.compute(&mut systems, Default::default()).expect("failed to run calculation");

    let keys_to_move = Labels::empty(vec!["center_type"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = Labels::empty(vec!["neighbor_type"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    let descriptor = descriptor.components_to_properties(&["o3_mu"]).unwrap();

    let keys_to_move = Labels::empty(vec!["o3_lambda"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.block_by_id(0);
    let array = block.values().to_array();

    let expected = &data::load_expected_values("spherical-expansion-pbc-values.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-6);
}

#[test]
fn gradients() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-gradients-input.json");
    let n_atoms = systems.iter().map(|s| s.size().unwrap()).sum();

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();

    let options = CalculationOptions {
        gradients: &["positions", "strain", "cell"],
        ..Default::default()
    };
    let descriptor = calculator.compute(&mut systems, options).expect("failed to run calculation");

    let keys_to_move = Labels::empty(vec!["center_type"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = Labels::empty(vec!["neighbor_type"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    let descriptor = descriptor.components_to_properties(&["o3_mu"]).unwrap();

    let keys_to_move = Labels::empty(vec!["o3_lambda"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.block_by_id(0);
    let gradient = block.gradient("positions").unwrap();
    let array = sum_gradients(n_atoms, gradient);

    let expected = &data::load_expected_values("spherical-expansion-positions-gradient.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-6);

    let gradient = block.gradient("strain").unwrap();
    let array = gradient.values().to_array();
    let expected = &data::load_expected_values("spherical-expansion-strain-gradient.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-6);

    let gradient = block.gradient("cell").unwrap();
    let array = gradient.values().to_array();
    let expected = &data::load_expected_values("spherical-expansion-cell-gradient.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-6);
}

fn sum_gradients(n_atoms: usize, gradients: TensorBlockRef<'_>) -> ArrayD<f64> {
    assert_eq!(gradients.samples().names(), &["sample", "system", "atom"]);
    let array = gradients.values().to_array();

    let mut sum = ArrayD::from_elem(vec![n_atoms, 3, gradients.properties().count()], 0.0);
    for ([_, _, atom], row) in gradients.samples().iter_fixed_size().zip(array.axis_iter(Axis(0))) {
        let mut slice = sum.slice_mut(s![atom.usize(), .., ..]);
        slice += &row;
    }

    sum
}
