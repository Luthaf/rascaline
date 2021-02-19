mod data;

use approx::assert_relative_eq;
use rascaline::{System, Calculator, Descriptor};

#[test]
fn spherical_expansion() {
    let (mut systems, parameters) = data::load_input("spherical-expansion-input.json");
    let mut systems = systems.iter_mut().map(|s| s as &mut dyn System).collect::<Vec<_>>();

    let mut descriptor = Descriptor::new();
    let mut calculator = Calculator::new("spherical_expansion", parameters).unwrap();
    calculator.compute(&mut systems, &mut descriptor, Default::default())
        .expect("failed to run calculation");

    let expected = data::load_expected_values("spherical-expansion-values.npy.gz");

    assert_eq!(descriptor.values.shape(), expected.shape());

    for i in 0..expected.nrows() {
        for j in 0..expected.ncols() {
            assert_relative_eq!(descriptor.values[[i, j]], expected[[i, j]], max_relative=1e-12);
        }
    }
}
