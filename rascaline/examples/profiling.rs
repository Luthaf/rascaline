use rascaline::{Calculator, Descriptor, System};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("expected a command line argument");

    // enable collection of profiling data
    time_graph::enable_data_collection(true);
    // clear any existing collected data
    time_graph::clear_collected_data();

    // run the calculation
    let descriptor = compute_soap(&path)?;

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
fn compute_soap(path: &str) -> Result<Descriptor, Box<dyn std::error::Error>> {
    let systems = rascaline::systems::read_from_file(path)?;
    let mut systems = systems.into_iter()
        .map(|s| Box::new(s) as Box<dyn System>)
        .collect::<Vec<_>>();

    let parameters = r#"{
        "cutoff": 5.0,
        "max_radial": 6,
        "max_angular": 4,
        "atomic_gaussian_width": 0.3,
        "gradients": true,
        "radial_basis": {
            "Gto": {}
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5}
        }
    }"#;

    let mut calculator = Calculator::new("soap_power_spectrum", parameters.to_owned())?;
    let mut descriptor = Descriptor::new();
    calculator.compute(&mut systems, &mut descriptor, Default::default())?;
    descriptor.densify(&["species_neighbor_1", "species_neighbor_2"], None)?;

    return Ok(descriptor);
}
