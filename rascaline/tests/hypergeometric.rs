use ndarray::{Array2, Array4, s};
use approx::assert_relative_eq;

use rascaline::calculators::soap::{
    HyperGeometricParameters,
    HyperGeometricSphericalExpansion
};

mod data;

#[derive(serde::Deserialize)]
struct HyperGeometricInput {
    all_rij: Vec<f64>,
    atomic_gaussian_constants: Vec<f64>,
    gto_gaussian_constants: Vec<f64>,
    max_angular: usize,
    max_radial: usize,
}

#[test]
fn hypergeometric() {
    let expected_values: Array4<f64> = data::load_expected_values("hypergeometric-values.npy.gz");
    let expected_gradients: Array4<f64> = data::load_expected_values("hypergeometric-gradients.npy.gz");

    let input = std::fs::read_to_string("tests/data/generated/hypergeometric-input.json").expect("failed to read file");
    let input: HyperGeometricInput = serde_json::from_str(&input).expect("failed to decode JSON");

    let hypergeometric = HyperGeometricSphericalExpansion::new(input.max_radial, input.max_angular);
    let shape = (input.max_radial, input.max_angular + 1);
    let mut values = Array2::from_elem(shape, 0.0);
    let mut gradients = Array2::from_elem(shape, 0.0);

    for (i_gaussian_constant, &atomic_gaussian_constant) in input.atomic_gaussian_constants.iter().enumerate() {
        let parameters = HyperGeometricParameters {
            atomic_gaussian_constant,
            gto_gaussian_constants: &input.gto_gaussian_constants
        };

        for (i_rij, &rij) in input.all_rij.iter().enumerate() {
            hypergeometric.compute(rij, parameters, values.view_mut(), Some(gradients.view_mut()));

            assert_relative_eq!(
                values, expected_values.slice(s![i_gaussian_constant, i_rij, .., ..]),
                max_relative=1e-11
            );

            assert_relative_eq!(
                gradients, expected_gradients.slice(s![i_gaussian_constant, i_rij, .., ..]),
                max_relative=1e-11
            );
        }
    }
}
