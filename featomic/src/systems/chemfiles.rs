use std::collections::HashMap;

use crate::Matrix3;
use crate::systems::UnitCell;

use super::{System, SimpleSystem};

impl<'a> From<&'a chemfiles::Frame> for Box<dyn System> {
    fn from(frame: &chemfiles::Frame) -> Self {
        let mut assigned_types = HashMap::new();
        let mut get_atomic_type = |atom: chemfiles::AtomRef| {
            let atomic_number = atom.atomic_number();
            if atomic_number == 0 {
                // use number assigned from the the atomic type, starting at 120
                // since that's larger than the number of elements in the
                // periodic table
                let new_type = 120 + assigned_types.len() as i32;
                *assigned_types.entry(atom.atomic_type()).or_insert(new_type)
            } else {
                atomic_number as i32
            }
        };
        let positions = frame.positions();

        let cell = if frame.cell().shape() == chemfiles::CellShape::Infinite {
            UnitCell::infinite()
        } else {
            // transpose since chemfiles is using columns for the cell vectors and
            // we want rows as cell vectors
            UnitCell::from(Matrix3::from(frame.cell().matrix()).transposed())
        };
        let mut system = SimpleSystem::new(cell);
        for i in 0..frame.size() {
            let atom = frame.atom(i);
            system.add_atom(get_atomic_type(atom), positions[i].into());
        }

        return Box::new(system) as Box<dyn System>;
    }
}

impl From<chemfiles::Frame> for Box<dyn System> {
    fn from(frame: chemfiles::Frame) -> Self {
        return (&frame).into();
    }
}
