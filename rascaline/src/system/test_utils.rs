use crate::{System, Vector3D};
use super::{UnitCell, SimpleSystem};

#[derive(Clone, Debug)]
pub struct SimpleSystems {
    pub(crate) systems: Vec<SimpleSystem>
}

impl SimpleSystems {
    pub fn boxed(self) -> Vec<Box<dyn System>> {
        self.systems.into_iter()
            .map(|s| Box::new(s) as Box<dyn System>)
            .collect()
    }
}

pub fn test_systems(names: &[&str]) -> SimpleSystems {
    let systems = names.iter().map(|&name| {
        match name {
            "methane" => get_methane(),
            "water" => get_water(),
            "CH" => get_ch(),
            _ => panic!("unknown test system {}", name)
        }
    }).collect();
    return SimpleSystems{ systems };
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
    system.add_atom(123456, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(1, Vector3D::new(0.0, 0.75545, -0.58895));
    system.add_atom(1, Vector3D::new(0.0, -0.75545, -0.58895));
    return system;
}

fn get_ch() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(10.0));
    system.add_atom(1, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(6, Vector3D::new(0.0, 1.2, 0.0));
    return system;
}
