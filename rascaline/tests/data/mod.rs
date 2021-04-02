#![allow(dead_code)]

use serde_json::Value;
use ndarray_npy::ReadNpyExt;
use flate2::read::GzDecoder;

use rascaline::{SimpleSystem, System, Matrix3, Vector3D};
use rascaline::system::UnitCell;

type HyperParameters = String;

pub fn load_calculator_input(path: &str) -> (Vec<Box<dyn System>>, HyperParameters) {
    let json = std::fs::read_to_string(&format!("tests/data/generated/{}", path))
        .expect("failed to read input file");

    let data: Value = serde_json::from_str(&json).expect("failed to parse JSON");
    let parameters = data["hyperparameters"].to_string();

    let mut systems = Vec::new();
    for system in data["systems"].as_array().expect("systems must be an array") {
        let cell = read_cell(&system["cell"]);
        let mut simple_system = SimpleSystem::new(cell);

        let species = system["species"].as_array().expect("species must be an array");
        let positions = system["positions"].as_array().expect("positions must be an array");

        for (species, position) in species.iter().zip(positions) {
            let species = species.as_u64().expect("species must be an integer") as usize;
            let position = position.as_array().expect("position must be an array");
            let position = Vector3D::new(
                position[0].as_f64().unwrap(),
                position[1].as_f64().unwrap(),
                position[2].as_f64().unwrap(),
            );

            simple_system.add_atom(species, position);
        }

        systems.push(Box::new(simple_system) as Box<dyn System>);
    }

    (systems, parameters)
}

fn read_cell(cell: &Value) -> UnitCell {
    let cell = cell.as_array().expect("cell must be an array");
    let matrix = Matrix3::new([
        [cell[0].as_f64().unwrap(), cell[1].as_f64().unwrap(), cell[2].as_f64().unwrap()],
        [cell[3].as_f64().unwrap(), cell[4].as_f64().unwrap(), cell[5].as_f64().unwrap()],
        [cell[6].as_f64().unwrap(), cell[7].as_f64().unwrap(), cell[8].as_f64().unwrap()],
    ]);

    if matrix == Matrix3::zero() {
        UnitCell::infinite()
    } else {
        UnitCell::from(matrix)
    }
}

pub fn load_expected_values<T: ReadNpyExt>(path: &str) -> T {
    let file = std::fs::File::open(&format!("tests/data/generated/{}", path))
        .expect("failed to open file");

    T::read_npy(GzDecoder::new(file)).expect("failed to convert data to ndarray")
}
