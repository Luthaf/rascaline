use std::path::Path;

use super::SimpleSystem;
use crate::Error;

#[cfg(feature = "chemfiles")]
impl From<chemfiles::Error> for Error {
    fn from(error: chemfiles::Error) -> Error {
        Error::Chemfiles(error.message)
    }
}

/// Read all structures in the file at the given `path` using
/// [chemfiles](https://chemfiles.org/), and convert them to `SimpleSystem`s.
///
/// This function can read all [formats supported by
/// chemfiles](https://chemfiles.org/chemfiles/latest/formats.html).
#[cfg(feature = "chemfiles")]
#[allow(clippy::needless_range_loop)]
pub fn read_from_file(path: impl AsRef<Path>) -> Result<Vec<SimpleSystem>, Error> {
    use std::collections::HashMap;
    use crate::Matrix3;
    use crate::systems::UnitCell;

    let mut systems = Vec::new();

    let mut trajectory = chemfiles::Trajectory::open(path, 'r')?;
    let mut frame = chemfiles::Frame::new();

    let mut assigned_species = HashMap::new();
    let mut get_species = |atom: chemfiles::AtomRef| {
        let atomic_number = atom.atomic_number();
        if atomic_number == 0 {
            // use number assigned from the the atomic type, starting at 120
            // since that's larger than the number of elements in the periodic
            // table
            let new_species = 120 + assigned_species.len();
            *assigned_species.entry(atom.atomic_type()).or_insert(new_species)
        } else {
            atomic_number as usize
        }
    };

    for _ in 0..trajectory.nsteps() {
        trajectory.read(&mut frame)?;

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
            system.add_atom(get_species(atom), positions[i].into());
        }

        systems.push(system);
    }

    return Ok(systems);
}

/// Read all structures in the file at the given `path` using
/// [chemfiles](https://chemfiles.org/), and convert them to `SimpleSystem`s.
///
/// This function can read all [formats supported by
/// chemfiles](https://chemfiles.org/chemfiles/latest/formats.html).
#[cfg(not(feature = "chemfiles"))]
pub fn read_from_file(_: impl AsRef<Path>) -> Result<Vec<SimpleSystem>, Error> {
    Err(Error::Chemfiles(
        "read_from_file is only available with the chemfiles feature enabled".into()
    ))
}

#[cfg(all(test, feature = "chemfiles"))]
mod tests {
    use std::path::PathBuf;
    use approx::assert_relative_eq;

    use crate::{System, Vector3D};
    use super::*;

    #[test]
    fn read() -> Result<(), Box<dyn std::error::Error>> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("benches");
        path.push("data");
        path.push("silicon_bulk.xyz");

        let systems = read_from_file(&path).unwrap();

        assert_eq!(systems.len(), 30);
        assert_eq!(systems[0].size()?, 54);
        assert_eq!(systems[0].species()?, [14; 54].as_ref());

        assert_relative_eq!(
            systems[0].positions()?[0],
            Vector3D::from([7.8554, 7.84887, 0.0188612])
        );

        let cell = systems[0].cell()?;
        assert_relative_eq!(cell.a(), 11.098535905469692);
        assert_relative_eq!(cell.b(), 11.098535905469692);
        assert_relative_eq!(cell.c(), 11.098535905469692);
        assert_relative_eq!(cell.alpha(), 60.0);
        assert_relative_eq!(cell.beta(), 60.0);
        assert_relative_eq!(cell.gamma(), 60.0);


        let matrix = cell.matrix();
        assert_eq!(matrix[0], [7.847849999999999, 0.0, 7.847849999999999]);
        assert_eq!(matrix[1], [7.847849999999999, 7.847849999999999, 0.0]);
        assert_eq!(matrix[2], [0.0, 7.847849999999999, 7.847849999999999]);

        Ok(())
    }
}
