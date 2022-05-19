use approx::assert_relative_eq;

use rascaline::Vector3D;
use rascaline::calculators::soap::{
    SphericalHarmonics,
    SphericalHarmonicsArray
};

mod data;

#[derive(serde::Deserialize)]
struct SphericalHarmonicsInput {
    directions: Vec<[f64; 3]>,
    max_angular: usize,
}

#[test]
fn spherical_harmonics() {
    let expected = data::load_expected_values("spherical-harmonics-values.npy.gz");

    let input = std::fs::read_to_string("tests/data/generated/spherical-harmonics-input.json").expect("failed to read file");
    let input: SphericalHarmonicsInput = serde_json::from_str(&input).expect("failed to decode JSON");

    let mut sph = SphericalHarmonics::new(input.max_angular);
    let mut values = SphericalHarmonicsArray::new(input.max_angular);

    for (i_direction, &direction) in input.directions.iter().enumerate() {
        let direction = Vector3D::from(direction);
        sph.compute(direction, &mut values, None);

        for l in 0..(input.max_angular as isize) {
            for (i_m, m) in (-l..=l).enumerate() {
                assert_relative_eq!(
                    values[[l, m]], expected[[i_direction, l as usize, i_m]],
                    epsilon=1e-11, max_relative=1e-11
                );
            }
        }
    }
}
