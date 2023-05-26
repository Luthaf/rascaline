use crate::{System, Vector3D};
use super::{UnitCell, SimpleSystem};

pub fn test_systems(names: &[&str]) -> Vec<Box<dyn System>> {
    return names.iter()
        .map(|&name| Box::new(test_system(name)) as Box<dyn System>)
        .collect();
}

pub fn test_system(name: &str) -> SimpleSystem {
    match name {
        "methane" => get_methane(),
        "ethanol" => get_ethanol(),
        "water" => get_water(),
        "CH" => get_ch(),
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

fn get_ethanol() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(6.0));
    system.add_atom(1, Vector3D::new(1.8853, -0.0401, 1.0854));
    system.add_atom(6, Vector3D::new(1.2699, -0.0477, 0.1772));
    system.add_atom(1, Vector3D::new(1.5840, 0.8007, -0.4449));
    system.add_atom(1, Vector3D::new(1.5089, -0.9636, -0.3791));
    system.add_atom(6, Vector3D::new(-0.2033, 0.0282, 0.5345));
    system.add_atom(1, Vector3D::new(-0.4993, -0.8287, 1.1714));
    system.add_atom(1, Vector3D::new(-0.4235, 0.9513, 1.1064));
    system.add_atom(8, Vector3D::new(-0.9394, 0.0157, -0.6674));
    system.add_atom(1, Vector3D::new(-1.8540, 0.0626, -0.4252));
    return system;
}
