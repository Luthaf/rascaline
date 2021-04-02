use rascaline::{Calculator, Descriptor, System};
use rascaline::system::{SimpleSystem, UnitCell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // load the systems from command line arguments
    let mut systems = Vec::new();
    for path in std::env::args().skip(1) {
        let file_content = std::fs::read_to_string(&path)?;
        // WARNING: this function only read the first step of the file
        let system = SimpleSystem::from_xyz(UnitCell::infinite(), &file_content);
        systems.push(Box::new(system) as Box<dyn System>);
    }

    // pass hyper-parameters as JSON
    let parameters = "{
        \"cutoff\": 5.0,
        \"max_radial\": 6,
        \"max_angular\": 4,
        \"atomic_gaussian_width\": 0.3,
        \"gradients\": false,
        \"radial_basis\": {
            \"GTO\": {}
        },
        \"cutoff_function\": {
            \"ShiftedCosine\": {\"width\": 0.5}
        }
    }";
    // create the calculator with its name and parameters
    let mut calculator = Calculator::new("spherical_expansion", parameters.to_owned())?;

    // create an empty descriptor
    let mut descriptor = Descriptor::new();

    // run the calculation using default options
    calculator.compute(&mut systems, &mut descriptor, Default::default())?;

    // Transform the descriptor to dense representation,
    // with one sample for each atom-centered environment
    descriptor.densify(vec!["species_neighbors"]);

    // you can now use descriptor.values as the
    // input of a machine learning algorithm

    Ok(())
}
