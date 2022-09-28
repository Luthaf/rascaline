use approx::assert_relative_eq;
use ndarray::{ArrayD, Axis, s};

use equistore::{LabelsBuilder, BasicBlock};

use rascaline::{Calculator, CalculationOptions};

mod data;

#[test]
fn values_no_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-values-input.json");

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    let mut descriptor = calculator.compute(&mut systems, Default::default()).expect("failed to run calculation");

    let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
    descriptor.keys_to_samples(&keys_to_move, true).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["species_neighbor"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    descriptor.components_to_properties(&["spherical_harmonics_m"]).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["spherical_harmonics_l"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.blocks()[0];
    let values = block.values();
    let array = values.data.as_array();

    let expected = &data::load_expected_values("spherical-expansion-values.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-9);
}

#[test]
fn values_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-pbc-values-input.json");

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    let mut descriptor = calculator.compute(&mut systems, Default::default()).expect("failed to run calculation");

    let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
    descriptor.keys_to_samples(&keys_to_move, true).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["species_neighbor"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    descriptor.components_to_properties(&["spherical_harmonics_m"]).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["spherical_harmonics_l"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.blocks()[0];
    let values = block.values();
    let array = values.data.as_array();

    let expected = &data::load_expected_values("spherical-expansion-pbc-values.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-9);
}

#[test]
fn gradients() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-gradients-input.json");
    let n_atoms = systems.iter().map(|s| s.size().unwrap()).sum();

    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();

    let options = CalculationOptions {
        gradients: &["positions", "cell"],
        ..Default::default()
    };
    let mut descriptor = calculator.compute(&mut systems, options).expect("failed to run calculation");

    let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
    descriptor.keys_to_samples(&keys_to_move, true).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["species_neighbor"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    descriptor.components_to_properties(&["spherical_harmonics_m"]).unwrap();
    let keys_to_move = LabelsBuilder::new(vec!["spherical_harmonics_l"]).finish();
    descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(descriptor.blocks().len(), 1);
    let block = &descriptor.blocks()[0];
    let gradient = block.gradient("positions").unwrap();
    let array = sum_gradients(n_atoms, gradient);

    let expected = &data::load_expected_values("spherical-expansion-positions-gradient.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-8);

    let gradient = block.gradient("cell").unwrap();
    let array = gradient.data.as_array();
    let expected = &data::load_expected_values("spherical-expansion-cell-gradient.npy.gz");
    assert_relative_eq!(array, expected, max_relative=1e-8);
}

fn sum_gradients(n_atoms: usize, gradients: &BasicBlock) -> ArrayD<f64> {
    assert_eq!(gradients.samples.names(), &["sample", "structure", "atom"]);
    let array = gradients.data.as_array();

    let mut sum = ArrayD::from_elem(vec![n_atoms, 3, gradients.properties.count()], 0.0);
    for ([_, _, atom], row) in gradients.samples.iter_fixed_size().zip(array.axis_iter(Axis(0))) {
        let mut slice = sum.slice_mut(s![atom.usize(), .., ..]);
        slice += &row;
    }

    sum
}
