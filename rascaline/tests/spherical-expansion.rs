use approx::assert_relative_eq;
use ndarray::{Array2, Array3, s};
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
    let n_atoms = systems.iter().map(|s| s.size().unwrap()).sum();

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected: Array3<f64> = data::load_expected_values("spherical-expansion-gradients.npy.gz");
    let gradients = sum_gradients(n_atoms, &descriptor);
    assert_eq!(gradients.shape(), expected.shape());
    assert_relative_eq!(gradients, expected, max_relative=1e-9);
}

fn sum_gradients(n_atoms: usize, descriptor: &Descriptor) -> Array3<f64> {
    let gradients = descriptor.gradients.as_ref().unwrap();
    let gradients_samples = descriptor.gradients_samples.as_ref().unwrap();
    assert_eq!(gradients_samples.names(), &["sample", "atom", "spatial"]);

    let mut sum = Array3::from_elem((n_atoms, 3, gradients.shape()[1]), 0.0);
    for (sample, gradient) in gradients_samples.iter().zip(gradients.rows()) {
        let neighbor = sample[1].usize();
        let spatial = sample[2].usize();
        let mut slice = sum.slice_mut(s![neighbor, spatial, ..]);
        slice += &gradient;
    }

    sum
}
