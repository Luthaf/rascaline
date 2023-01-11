use equistore::Labels;
use rascaline::{Calculator, System, CalculationOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // load the systems from command line argument
    let path = std::env::args().nth(1).expect("expected a command line argument");
    let systems = rascaline::systems::read_from_file(path)?;
    // transform systems into a vector of trait objects (`Vec<Box<dyn System>>`)
    let mut systems = systems.into_iter()
        .map(|s| Box::new(s) as Box<dyn System>)
        .collect::<Vec<_>>();

    // pass hyper-parameters as JSON
    let parameters = r#"{
        "cutoff": 5.0,
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {}
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5}
        }
    }"#;
    // create the calculator with its name and parameters
    let mut calculator = Calculator::new("soap_power_spectrum", parameters.to_owned())?;

    // create the options for a single calculation, here we request the
    // calculation of gradients with respect to positions
    let options = CalculationOptions {
        gradients: &["positions"],
        ..Default::default()
    };

    // run the calculation
    let descriptor = calculator.compute(&mut systems, options)?;

    // Transform the descriptor to dense representation, with one sample for
    // each atom-centered environment, and all neighbor species part of the
    // properties
    let keys_to_move = Labels::empty(vec!["species_center"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, /* sort_samples */ true)?;

    let keys_to_move = Labels::empty(vec!["species_neighbor_1", "species_neighbor_2"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, /* sort_samples */ true)?;

    // descriptor now contains a single block, which can be used as the input
    // to standard ML algorithms
    let values = descriptor.block_by_id(0).values();
    println!("the shape of the values is {:?}", values.data.as_array().shape());

    Ok(())
}
