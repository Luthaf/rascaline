use approx::assert_relative_eq;
use ndarray::Array2;
use rascaline::{Calculator, Descriptor};

mod data;

#[test]
fn values() {
    let (mut systems, parameters) = data::load_calculator_input("soap-power-spectrum-values-input.json");

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("soap_power_spectrum", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array2<f64> = data::load_expected_values("soap-power-spectrum-values.npy.gz");
    assert_eq!(descriptor.values.shape(), expected.shape());
    assert_relative_eq!(descriptor.values, expected, max_relative=1e-9);

    descriptor.densify(&["species_neighbor_1", "species_neighbor_2"]);
    let expected: Array2<f64> = data::load_expected_values("soap-power-spectrum-dense-values.npy.gz");
    assert_eq!(descriptor.values.shape(), expected.shape());
    assert_relative_eq!(descriptor.values, expected, max_relative=1e-9);
}

#[test]
fn gradients() {
    let (mut systems, parameters) = data::load_calculator_input("soap-power-spectrum-gradients-input.json");

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("soap_power_spectrum", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array2<f64> = data::load_expected_values("soap-power-spectrum-gradients.npy.gz");

    let gradients = descriptor.gradients.as_ref().unwrap();
    assert_eq!(gradients.shape(), expected.shape());
    assert_relative_eq!(*gradients, expected, max_relative=1e-9);

    descriptor.densify(&["species_neighbor_1", "species_neighbor_2"]);
    let expected: Array2<f64> = data::load_expected_values("soap-power-spectrum-dense-gradients.npy.gz");

    let gradients = descriptor.gradients.as_ref().unwrap();
    assert_eq!(gradients.shape(), expected.shape());
    assert_relative_eq!(*gradients, expected, max_relative=1e-9);
}
