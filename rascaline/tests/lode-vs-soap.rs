use approx::assert_relative_eq;

use rascaline::systems::{System, SimpleSystem, UnitCell};
use rascaline::{Vector3D, Calculator};

use equistore::LabelsBuilder;


#[test]
fn lode_vs_soap() {
    // simple tetramer of Oxygen
    let mut system = SimpleSystem::new(UnitCell::cubic(16.0));
    system.add_atom(8, Vector3D::new(1.0, 1.0, 1.0));
    system.add_atom(8, Vector3D::new(2.0, 1.0, 1.0));
    system.add_atom(8, Vector3D::new(2.0, 2.2, 1.0));
    system.add_atom(8, Vector3D::new(2.3, 2.0, 1.5));

    let mut systems = vec![Box::new(system) as Box<dyn System>];

    let lode_parameters = r#"{
        "cutoff": 6.0,
        "k_cutoff": 14.0,
        "max_radial": 6,
        "max_angular": 6,
        "atomic_gaussian_width": 0.3,
        "potential_exponent": 0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-8}}
    }"#;

    let soap_parameters = r#"{
        "cutoff": 6.0,
        "max_radial": 6,
        "max_angular": 6,
        "center_atom_weight": 1.0,
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-8}},
        "cutoff_function": {"Step": {}}
    }"#;

    let mut lode_calculator = Calculator::new(
        "lode_spherical_expansion",
        lode_parameters.into(),
    ).unwrap();

    let mut soap_calculator = Calculator::new(
        "spherical_expansion",
        soap_parameters.into(),
    ).unwrap();

    let mut lode_descriptor = lode_calculator.compute(&mut systems, Default::default()).unwrap();
    let mut soap_descriptor = soap_calculator.compute(&mut systems, Default::default()).unwrap();

    let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
    lode_descriptor.keys_to_samples(&keys_to_move, true).unwrap();
    soap_descriptor.keys_to_samples(&keys_to_move, true).unwrap();

    let keys_to_move = LabelsBuilder::new(vec!["species_neighbor"]).finish();
    lode_descriptor.keys_to_properties(&keys_to_move, true).unwrap();
    soap_descriptor.keys_to_properties(&keys_to_move, true).unwrap();

    // TODO: check all blocks
    assert_relative_eq!(
        lode_descriptor.blocks()[0].values().data.as_array(),
        soap_descriptor.blocks()[0].values().data.as_array(),
        max_relative=1e-4
    );
}
