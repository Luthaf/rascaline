use std::path::PathBuf;


use approx::assert_relative_eq;
use ndarray::{ArrayD, Axis, s};

use metatensor::{Labels, TensorBlockRef};

use rascaline::{Calculator, CalculationOptions};

mod data;

#[test]
fn values() {
    for potential_exponent in [1, 2, 3, 4, 5, 6] {
        let mut path = PathBuf::from("lode-spherical-expansion");
        path.push(format!("potential_exponent-{}", potential_exponent));
        path.push("values-input.json");

        let (mut systems, parameters) = data::load_calculator_input(path);

        let mut calculator = Calculator::new("lode_spherical_expansion", parameters).unwrap();
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

        let mut path = PathBuf::from("lode-spherical-expansion");
        path.push(format!("potential_exponent-{}", potential_exponent));
        path.push("values.npy.gz");
        let expected = &data::load_expected_values(path);

        assert_relative_eq!(array, expected, max_relative=1e-6, epsilon=1e-13);
    }
}

#[test]
fn gradients() {
    for potential_exponent in [1, 2, 3, 4, 5, 6] {
        let mut path = PathBuf::from("lode-spherical-expansion");
        path.push(format!("potential_exponent-{}", potential_exponent));
        path.push("gradients-input.json");

        let (mut systems, parameters) = data::load_calculator_input(path);
        let n_atoms = systems.iter().map(|s| s.size().unwrap()).sum();

        let mut calculator = Calculator::new("lode_spherical_expansion", parameters).unwrap();

        let options = CalculationOptions {
            gradients: &["positions"],
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

        let mut path = PathBuf::from("lode-spherical-expansion");
        path.push(format!("potential_exponent-{}", potential_exponent));
        path.push("positions-gradient.npy.gz");

        let expected = &data::load_expected_values(path);
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }
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
