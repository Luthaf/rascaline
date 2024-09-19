use metatensor::Labels;
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
        "cutoff": {
            "radius": 5.0,
            "smoothing": {
                "type": "ShiftedCosine",
                "width": 0.5
            }
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"type": "Gto", "max_radial": 6}
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
    // each atom-centered environment, and the neighbor atomic types part of the
    // properties
    let keys_to_move = Labels::empty(vec!["center_type"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, /* sort_samples */ true)?;

    let keys_to_move = Labels::empty(vec!["neighbor_1_type", "neighbor_2_type"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, /* sort_samples */ true)?;

    // descriptor now contains a single block, which can be used as the input
    // to standard ML algorithms
    let values = descriptor.block_by_id(0).values().to_array();
    println!("SOAP representation shape: {:?}", values.shape());

    Ok(())
}
