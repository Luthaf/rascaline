use super::{UnitCell, System, Vector3D};

struct SimpleSystem {
    cell: UnitCell,
    types: Vec<usize>,
    positions: Vec<Vector3D>,
}

impl System for SimpleSystem {
    fn natoms(&self) -> usize {
        self.positions.len()
    }

    fn positions(&self) -> &[Vector3D] {
        &self.positions
    }

    fn types(&self) -> &[usize] {
        &self.types
    }

    fn cell(&self) -> UnitCell {
        self.cell
    }
}

pub fn test_system(name: &str) -> Box<dyn System> {
    match name {
        "methane" => Box::new(get_methane()),
        _ => panic!("unknown test system {}", name)
    }
}

fn get_methane() -> SimpleSystem {
    let positions = vec![
        Vector3D::new(0.0, 0.0, 0.0),
        Vector3D::new(0.5288, 0.1610, 0.9359),
        Vector3D::new(0.2051, 0.8240, -0.6786),
        Vector3D::new(0.3345, -0.9314, -0.4496),
        Vector3D::new(-1.0685, -0.0537, 0.1921),
    ];
    SimpleSystem {
        cell: UnitCell::cubic(10.0),
        positions: positions,
        types: vec![6, 1, 1, 1, 1],
    }
}
