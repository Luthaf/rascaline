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

    // reshape the array features from `lmn` (produced by the current version of
    // the code) to `nlm` (used by the code at the time when the array was
    // saved). This reshaping code should be removed next time we have to update
    // the saved data.
    let n_samples = descriptor.values.shape()[0];
    let n_features = descriptor.values.shape()[1];

    let n_radial = 8;
    let n_angular = n_features / n_radial;

    let mut values = descriptor.values.to_shape((n_samples, n_angular, n_radial)).unwrap();
    values.swap_axes(1, 2);
    let values = values.to_shape((n_samples, n_features)).unwrap();
    // end of reshaping code

    assert_relative_eq!(values, expected, max_relative=1e-9);
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

    // reshape the array features from `lmn` (produced by the current version of
    // the code) to `nlm` (used by the code at the time when the array was
    // saved). This reshaping code should be removed next time we have to update
    // the saved data.
    let n_samples = descriptor.values.shape()[0];
    let n_features = descriptor.values.shape()[1];

    let n_radial = 6;
    let n_angular = n_features / n_radial;

    let mut values = descriptor.values.to_shape((n_samples, n_angular, n_radial)).unwrap();
    values.swap_axes(1, 2);
    let values = values.to_shape((n_samples, n_features)).unwrap();
    // end of reshaping code

    assert_relative_eq!(values, expected, max_relative=1e-9);
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

    // reshape the array features from `lmn` (produced by the current version of
    // the code) to `nlm` (used by the code at the time when the array was
    // saved). This reshaping code should be removed next time we have to update
    // the saved data.
    let n_features = gradients.shape()[2];

    let n_radial = 4;
    let n_angular = n_features / n_radial;

    let mut gradients = gradients.to_shape((n_atoms, 3, n_angular, n_radial)).unwrap();
    gradients.swap_axes(2, 3);
    let gradients = gradients.to_shape((n_atoms, 3, n_features)).unwrap();
    // end of reshaping code

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
