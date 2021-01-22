use crate::{System, Vector3D};
use super::{UnitCell, SimpleSystem};

pub struct SimpleSystems {
    systems: Vec<SimpleSystem>
}

impl SimpleSystems {
    pub fn get(&mut self) -> Vec<&mut dyn System> {
        let mut references = Vec::new();
        for system in &mut self.systems {
            references.push(system as &mut dyn System)
        }
        return references;
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
    SimpleSystem::from_xyz(UnitCell::cubic(10.0), "5

        C 5.0000 5.0000 5.0000
        H 5.5288 5.1610 5.9359
        H 5.2051 5.8240 4.3214
        H 5.3345 4.0686 4.5504
        H 3.9315 4.9463 5.1921"
    )
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
