use super::{UnitCell, System, Vector3D, Pair};

pub struct CrappyNeighborsList {
    cutoff: f64,
    pairs: Vec<Pair>,
}

impl CrappyNeighborsList {
    pub fn new<S: System + ?Sized>(system: &S, cutoff: f64) -> CrappyNeighborsList {
        let cutoff2 = cutoff * cutoff;
        let cell = system.cell();
        let natoms = system.size();
        let positions = system.positions();

        let mut pairs = vec![];
        // crappy implementation, looping over all atoms in the system
        for i in 0..natoms {
            for j in (i + 1)..natoms {
                let d2 = cell.distance2(&positions[i], &positions[j]);
                if d2 < cutoff2 {
                    if i < j {
                        pairs.push(Pair{ first: i, second: j, distance: d2.sqrt()});
                    } else {
                        pairs.push(Pair{ first: j, second: i, distance: d2.sqrt()});
                    }
                }
            }
        }

        return CrappyNeighborsList {
            cutoff: cutoff,
            pairs: pairs,
        };
    }
}

struct SimpleSystem {
    cell: UnitCell,
    species: Vec<usize>,
    positions: Vec<Vector3D>,
    neighbors: Option<CrappyNeighborsList>,
}

impl System for SimpleSystem {
    fn size(&self) -> usize {
        self.species.len()
    }

    fn positions(&self) -> &[Vector3D] {
        &self.positions
    }

    fn species(&self) -> &[usize] {
        &self.species
    }

    fn cell(&self) -> UnitCell {
        self.cell
    }

    fn compute_neighbors(&mut self, cutoff: f64) {
        // re-use precompute NL is possible
        if let Some(ref nl) = self.neighbors {
            if nl.cutoff == cutoff {
                return;
            }
        }

        self.neighbors = Some(CrappyNeighborsList::new(self, cutoff));
    }

    fn pairs(&self) -> &[Pair] {
        &self.neighbors.as_ref().expect("neighbor list is not initialized").pairs
    }
}

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

pub fn test_systems(names: Vec<&str>) -> SimpleSystems {
    let systems = names.iter().map(|&name| {
        match name {
            "methane" => get_methane(),
            "water" => get_water(),
            "ch" => get_ch(),
            _ => panic!("unknown test system {}", name)
        }
    }).collect();
    return SimpleSystems{ systems };
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
        species: vec![6, 1, 1, 1, 1],
        neighbors: None,
    }
}

fn get_water() -> SimpleSystem {
    let positions = vec![
        Vector3D::new(0.0, 0.0, 0.0),
        Vector3D::new(0.0, 0.75545, -0.58895),
        Vector3D::new(0.0, -0.75545, -0.58895),
    ];
    SimpleSystem {
        cell: UnitCell::cubic(10.0),
        positions: positions,
        // species do not have to be atomic number
        species: vec![123456, 1, 1],
        neighbors: None,
    }
}

fn get_ch() -> SimpleSystem {
    SimpleSystem {
        cell: UnitCell::cubic(10.0),
        positions: vec![Vector3D::new(0.0, 0.0, 0.0), Vector3D::new(0.0, 1.2, 0.0)],
        species: vec![1, 6],
        neighbors: None,
    }
}
