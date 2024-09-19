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
        (2, 0)
    } else {
        (5, 2)
    };

    let lode_parameters = format!(r#"{{
        "k_cutoff": 16.0,
        "density": {{
            "type": "SmearedPowerLaw",
            "smearing": 0.3,
            "exponent": 0
        }},
        "basis": {{
            "type": "TensorProduct",
            "max_angular": {},
            "radial": {{"max_radial": {}, "type": "Gto", "radius": 3.0}},
            "spline_accuracy": null
        }}
    }}"#, max_angular, max_radial);

    let soap_parameters = format!(r#"{{
        "cutoff": {{
            "radius": 3.0,
            "smoothing": {{ "type": "Step" }}
        }},
        "density": {{
            "type": "Gaussian",
            "width": 0.3
        }},
        "basis": {{
            "type": "TensorProduct",
            "max_angular": {},
            "radial": {{"max_radial": {}, "type": "Gto"}},
            "spline_accuracy": null
        }}
    }}"#, max_angular, max_radial);


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
            lode.values().to_array(),
            soap.values().to_array(),
            max_relative=1e-4
        );
    }
}
