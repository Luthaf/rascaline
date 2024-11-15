use metatensor::{TensorMap, Labels};
use featomic::{Calculator, System, CalculationOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("expected a command line argument");

    // enable collection of profiling data
    time_graph::enable_data_collection(true);
    // clear any existing collected data
    time_graph::clear_collected_data();

    // run the calculation
    let _descriptor = compute_soap(&path)?;

    // get the call graph and display it
    let graph = time_graph::get_full_graph();
    // (this requires the "table" feature for the time_graph crate)
    println!("{}", graph.as_short_table());

    // also available for saving profiling data to the disk & future analysis
    // (this requires the "json" feature for the time_graph crate)
    println!("{}", graph.as_json());

    Ok(())
}


/// Compute SOAP power spectrum, this is the same code as the 'compute-soap'
/// example
fn compute_soap(path: &str) -> Result<TensorMap, Box<dyn std::error::Error>> {
    let systems = featomic::systems::read_from_file(path)?;
    let mut systems = systems.into_iter()
        .map(|s| Box::new(s) as Box<dyn System>)
        .collect::<Vec<_>>();

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

    let descriptor = time_graph::spanned!("Full calculation", {
        let mut calculator = Calculator::new("soap_power_spectrum", parameters.to_owned())?;

        let options = CalculationOptions {
            gradients: &["positions"],
            ..Default::default()
        };
        calculator.compute(&mut systems, options)?
    });

    let keys_to_move = Labels::empty(vec!["center_type"]);
    let descriptor = descriptor.keys_to_samples(&keys_to_move, /* sort_samples */ true)?;

    let keys_to_move = Labels::empty(vec!["neighbor_1_type", "neighbor_2_type"]);
    let descriptor = descriptor.keys_to_properties(&keys_to_move, /* sort_samples */ true)?;

    Ok(descriptor)
}
