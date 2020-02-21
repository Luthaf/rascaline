use super::{System, NeighborsList};

pub struct CrappyNeighborsList {
    neighbors: Vec<Vec<usize>>,
}

impl CrappyNeighborsList {
    pub fn new<S: System + ?Sized>(system: &S, cutoff: f64) -> CrappyNeighborsList {
        let cutoff2 = cutoff * cutoff;
        let cell = system.cell();
        let natoms = system.natoms();
        let positions = system.positions();

        let mut neighbors = vec![Vec::new(); natoms];
        // crappy implementation, looping over all atoms in the system
        for i in 0..natoms {
            for j in (i + 1)..natoms {
                if cell.distance2(&positions[i], &positions[j]) < cutoff2 {
                    if i < j {
                        neighbors[i].push(j);
                    } else {
                        neighbors[j].push(i);
                    }
                }
            }
        }
        return CrappyNeighborsList { neighbors }
    }
}

impl NeighborsList for CrappyNeighborsList {
    fn natoms(&self) -> usize {
        self.neighbors.len()
    }

    fn foreach_pairs(&self, function: &mut dyn FnMut(usize, usize)) {
        for (i, neighbors) in self.neighbors.iter().enumerate() {
            for &j in neighbors {
                function(i, j);
            }
        }
    }
}
