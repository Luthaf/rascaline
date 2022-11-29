use approx::assert_relative_eq;

use rascaline::systems::{System, SimpleSystem, UnitCell};
use rascaline::{Vector3D, Calculator};


#[test]
fn lode_vs_soap() {
    // simple tetramer of Oxygen
    let mut system = SimpleSystem::new(UnitCell::cubic(20.0));
    system.add_atom(8, Vector3D::new(1.0, 1.0, 1.0));
    system.add_atom(8, Vector3D::new(2.0, 1.0, 1.0));
    system.add_atom(8, Vector3D::new(2.0, 2.2, 1.0));
    system.add_atom(8, Vector3D::new(2.3, 2.0, 1.5));

    let mut systems = vec![Box::new(system) as Box<dyn System>];

    // reduce max_radial/max_angular for debug builds to make this test faster
    let (max_radial, max_angular) = if cfg!(debug_assertions) {
        (3, 0)
    } else {
        (6, 2)
    };

    let lode_parameters = format!(r#"{{
        "cutoff": 3.0,
        "k_cutoff": 16.0,
        "max_radial": {},
        "max_angular": {},
        "center_atom_weight": 1.0,
        "atomic_gaussian_width": 0.3,
        "potential_exponent": 0,
        "radial_basis": {{"Gto": {{"splined_radial_integral": false}}}}
    }}"#, max_radial, max_angular);

    let soap_parameters = format!(r#"{{
        "cutoff": 3.0,
        "max_radial": {},
        "max_angular": {},
        "center_atom_weight": 1.0,
        "atomic_gaussian_width": 0.3,
        "radial_basis": {{"Gto": {{"splined_radial_integral": false}}}},
        "cutoff_function": {{"Step": {{}}}}
    }}"#, max_radial, max_angular);

    let mut lode_calculator = Calculator::new(
        "lode_spherical_expansion",
        lode_parameters,
    ).unwrap();

    let mut soap_calculator = Calculator::new(
        "spherical_expansion",
        soap_parameters,
    ).unwrap();

    let lode_descriptor = lode_calculator.compute(&mut systems, Default::default()).unwrap();
    let soap_descriptor = soap_calculator.compute(&mut systems, Default::default()).unwrap();
    assert_eq!(lode_descriptor.keys(), soap_descriptor.keys());

    for (soap, lode) in lode_descriptor.blocks().iter().zip(soap_descriptor.blocks()) {
        assert_relative_eq!(
            lode.values().data.as_array(),
            soap.values().data.as_array(),
            max_relative=1e-4
        );
    }
}
