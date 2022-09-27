use crate::{System, Vector3D, Matrix3};
use super::{UnitCell, SimpleSystem};

pub fn test_systems(names: &[&str]) -> Vec<Box<dyn System>> {
    return names.iter()
        .map(|&name| Box::new(test_system(name)) as Box<dyn System>)
        .collect();
}

pub fn test_system(name: &str) -> SimpleSystem {
    match name {
        "methane" => get_methane(),
        "water" => get_water(),
        "CH" => get_ch(),
        "NaCl" => get_nacl(),
        "CsCl" => get_cscl(),
        "ZnS" => get_zns(),
        "ZnSO4" => get_znso4(),
        _ => panic!("unknown test system {}", name)
    }
}

fn get_methane() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(5.0));
    system.add_atom(6, Vector3D::new(5.0000, 5.0000, 5.0000));
    system.add_atom(1, Vector3D::new(5.5288, 5.1610, 5.9359));
    system.add_atom(1, Vector3D::new(5.2051, 5.8240, 4.3214));
    system.add_atom(1, Vector3D::new(5.3345, 4.0686, 4.5504));
    system.add_atom(1, Vector3D::new(3.9315, 4.9463, 5.1921));
    return system;
}

fn get_water() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(10.0));
    // species do not have to be atomic number
    system.add_atom(-42, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(1, Vector3D::new(0.0, 0.75545, -0.58895));
    system.add_atom(1, Vector3D::new(0.0, -0.75545, -0.58895));
    return system;
}

fn get_ch() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(10.0));
    system.add_atom(6, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(1, Vector3D::new(0.0, 1.2, 0.0));
    return system;
}

/// NaCl structure
/// Using a primitive unit cell, the distance between the
/// closest Na-Cl pair is exactly 1. The cubic unit cell
/// in these units would have a length of 2.
fn get_nacl() -> SimpleSystem {
    let cell = Matrix3::new([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(11, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(17, Vector3D::new(1.0, 0.0, 0.0));
    return system;
}

/// CsCl structure
/// This structure is simple since the primitive unit cell
/// is just the usual cubic cell with side length set to one.
fn get_cscl() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(1.0));
    system.add_atom(17, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(55, Vector3D::new(0.5, 0.5, 0.5));
    return system;
}

/// ZnS (zincblende) structure
/// As for NaCl, a primitive unit cell is used which makes
/// the lattice parameter of the cubic cell equal to 2.
/// In these units, the closest Zn-S distance is sqrt(3)/2.
fn get_zns() -> SimpleSystem {
    let cell = Matrix3::new([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(16, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(30, Vector3D::new(0.5, 0.5, 0.5));
    return system;
}


/// ZnS (O4) in wurtzite structure (triclinic cell)
fn get_znso4() -> SimpleSystem {
    let u = 3. / 8.;
    let c = f64::sqrt(1. / u);
    let cell = Matrix3::new([[0.5, -0.5 * f64::sqrt(3.0), 0.0], [0.5, 0.5 * f64::sqrt(3.0), 0.0], [0.0, 0.0, c]]);
    let mut system = SimpleSystem::new(UnitCell::from(cell));
    system.add_atom(16, Vector3D::new(0.5, 0.5 / f64::sqrt(3.0), 0.0));
    system.add_atom(30, Vector3D::new(0.5, 0.5 / f64::sqrt(3.0), u * c));
    system.add_atom(16, Vector3D::new(0.5, -0.5 / f64::sqrt(3.0), 0.5 * c));
    system.add_atom(30, Vector3D::new(0.5, -0.5 / f64::sqrt(3.0), (0.5 + u) * c));
    return system;
}
