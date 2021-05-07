use approx::assert_relative_eq;
use ndarray::Array2;
use rascaline::{Calculator, Descriptor};

mod data;

#[test]
fn values_no_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-values-input.json");

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array2<f64> = data::load_expected_values("spherical-expansion-values.npy.gz");
    assert_eq!(descriptor.values.shape(), expected.shape());
    assert_relative_eq!(descriptor.values, expected, max_relative=1e-9);
}

#[test]
fn values_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-pbc-values-input.json");

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array2<f64> = data::load_expected_values("spherical-expansion-pbc-values.npy.gz");
    assert_eq!(descriptor.values.shape(), expected.shape());
    assert_relative_eq!(descriptor.values, expected, max_relative=1e-9);
}

#[test]
fn gradients_no_pbc() {
    let (mut systems, parameters) = data::load_calculator_input("spherical-expansion-gradients-input.json");

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array2<f64> = data::load_expected_values("spherical-expansion-gradients.npy.gz");
    let gradients = descriptor.gradients.unwrap();
    assert_eq!(gradients.shape(), expected.shape());
    assert_relative_eq!(gradients, expected, max_relative=1e-9);
}
