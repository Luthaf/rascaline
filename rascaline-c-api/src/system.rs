use std::os::raw::{c_char, c_void};
use std::ffi::CStr;

use rascaline::types::{Vector3D, Matrix3};
use rascaline::systems::{SimpleSystem, Pair, UnitCell};
use rascaline::{Error, System};

use super::{catch_unwind, rascal_status_t};

/// Pair of atoms coming from a neighbor list
#[repr(C)]
pub struct rascal_pair_t {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
    /// distance between the two atoms
    pub distance: f64,
    /// vector from the first atom to the second atom, wrapped inside the unit
    /// cell as required by periodic boundary conditions.
    pub vector: [f64; 3],
}

/// A `rascal_system_t` deals with the storage of atoms and related information,
/// as well as the computation of neighbor lists.
///
/// This struct contains a manual implementation of a virtual table, allowing to
/// implement the rust `System` trait in C and other languages. Speaking in Rust
/// terms, `user_data` contains a pointer (analog to `Box<Self>`) to the struct
/// implementing the `System` trait; and then there is one function pointers
/// (`Option<unsafe extern fn(XXX)>`) for each function in the `System` trait.
///
/// A new implementation of the System trait can then be created in any language
/// supporting a C API (meaning any language for our purposes); by correctly
/// setting `user_data` to the actual data storage, and setting all function
/// pointers to the correct functions. For an example of code doing this, see
/// the `SystemBase` class in the Python interface to rascaline.

// Function pointers have type `Option<unsafe extern fn(XXX)>`, where `Option`
// ensure that the `impl System for rascal_system_t` is forced to deal with the
// function pointer potentially being NULL. `unsafe` is required since these
// function come from another language and are not checked by the Rust compiler.
// Finally `extern` defaults to `extern "C"`, setting the ABI of the function to
// the default C ABI on the current system.
#[repr(C)]
pub struct rascal_system_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    user_data: *mut c_void,
    /// This function should set `*size` to the number of atoms in this system
    size: Option<unsafe extern fn(user_data: *const c_void, size: *mut usize)>,
    /// This function should set `*species` to a pointer to the first element of
    /// a contiguous array containing the atomic species of each atom in the
    /// system. Different atomic species should be identified with a different
    /// value. These values are usually the atomic number, but don't have to be.
    /// The array should contain `rascal_system_t::size()` elements.
    species: Option<unsafe extern fn(user_data: *const c_void, species: *mut *const usize)>,
    /// This function should set `*positions` to a pointer to the first element
    /// of a contiguous array containing the atomic cartesian coordinates.
    /// `positions[0], positions[1], positions[2]` must contain the x, y, z
    /// cartesian coordinates of the first atom, and so on.
    positions: Option<unsafe extern fn(user_data: *const c_void, positions: *mut *const f64)>,
    /// This function should write the unit cell matrix in `cell`, which have
    /// space for 9 values. The cell should be written in row major order, i.e.
    /// `ax ay az bx by bz cx cy cz`, where a/b/c are the unit cell vectors.
    cell: Option<unsafe extern fn(user_data: *const c_void, cell: *mut f64)>,
    /// This function should compute the neighbor list with the given cutoff,
    /// and store it for later access using `pairs` or `pairs_containing`.
    compute_neighbors: Option<unsafe extern fn(user_data: *mut c_void, cutoff: f64)>,
    /// This function should set `*pairs` to a pointer to the first element of a
    /// contiguous array containing all pairs in this system; and `*count` to
    /// the size of the array/the number of pairs.
    ///
    /// This list of pair should only contain each pair once (and not twice as
    /// `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
    /// contains pairs where the distance between atoms is actually bellow the
    /// cutoff passed in the last call to `compute_neighbors`. This function is
    /// only valid to call after a call to `compute_neighbors`.
    pairs: Option<unsafe extern fn(user_data: *const c_void, pairs: *mut *const rascal_pair_t, count: *mut usize)>,
    /// This function should set `*pairs` to a pointer to the first element of a
    /// contiguous array containing all pairs in this system containing the atom
    /// with index `center`; and `*count` to the size of the array/the number of
    /// pairs.
    ///
    /// The same restrictions on the list of pairs as `rascal_system_t::pairs`
    /// applies, with the additional condition that the pair `i-j` should be
    /// included both in the return of `pairs_containing(i)` and
    /// `pairs_containing(j)`.
    pairs_containing: Option<unsafe extern fn(user_data: *const c_void, center: usize, pairs: *mut *const rascal_pair_t, count: *mut usize)>,
}

impl<'a> System for &'a mut rascal_system_t {
    fn size(&self) -> Result<usize, Error> {
        let mut value = 0;
        let function = self.size.ok_or_else(|| Error::System(
            "rascal_system_t.size function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, &mut value);
        }
        return Ok(value);
    }

    fn species(&self) -> Result<&[usize], Error> {
        let mut ptr = std::ptr::null();
        let function = self.species.ok_or_else(|| Error::System(
            "rascal_system_t.species function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, &mut ptr);
        }

        let size = self.size()?;
        if ptr.is_null() && size != 0 {
            return Err(Error::System(
                "rascal_system_t.species returned a NULL pointer with non zero size".into()
            ));
        }

        unsafe {
            return Ok(std::slice::from_raw_parts(ptr, self.size()?));
        }
    }

    fn positions(&self) -> Result<&[Vector3D], Error> {
        let mut ptr = std::ptr::null();
        let function = self.positions.ok_or_else(|| Error::System(
            "rascal_system_t.positions function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, &mut ptr);
        }

        let size = self.size()?;
        if ptr.is_null() && size != 0 {
            return Err(Error::System(
                "rascal_system_t.positions returned a NULL pointer with non zero size".into()
            ));
        }

        unsafe {
            return Ok(std::slice::from_raw_parts(ptr.cast(), self.size()?));
        }
    }

    fn cell(&self) -> Result<UnitCell, Error> {
        let mut value = [[0.0; 3]; 3];
        let function = self.cell.ok_or_else(|| Error::System(
            "rascal_system_t.cell function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, &mut value[0][0]);
        }

        let matrix = Matrix3::from(value);
        if matrix == Matrix3::zero() {
            Ok(UnitCell::infinite())
        } else {
            Ok(UnitCell::from(matrix))
        }
    }

    fn compute_neighbors(&mut self, cutoff: f64) -> Result<(), Error> {
        let function = self.compute_neighbors.ok_or_else(|| Error::System(
            "rascal_system_t.compute_neighbors function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, cutoff);
        }
        Ok(())
    }

    fn pairs(&self) -> Result<&[Pair], Error> {
        let mut ptr = std::ptr::null();
        let mut count = 0;

        let function = self.pairs.ok_or_else(|| Error::System(
            "rascal_system_t.pairs function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, &mut ptr, &mut count);
        }

        if ptr.is_null() && count != 0 {
            return Err(Error::System(
                "rascal_system_t.positions returned a NULL pointer with non zero size".into()
            ));
        }
        unsafe {
            // SAFETY: ptr is non null, and Pair / rascal_pair_t have the same layout
            return Ok(std::slice::from_raw_parts(ptr.cast(), count));
        }
    }

    fn pairs_containing(&self, center: usize) -> Result<&[Pair], Error> {
        let mut ptr = std::ptr::null();
        let mut count = 0;

        let function = self.pairs_containing.ok_or_else(|| Error::System(
            "rascal_system_t.pairs_containing function is NULL".into()
        ))?;
        unsafe {
            function(self.user_data, center, &mut ptr, &mut count);
        }

        if ptr.is_null() && count != 0 {
            return Err(Error::System(
                "rascal_system_t.positions returned a NULL pointer with non zero size".into()
            ));
        }
        unsafe {
            // SAFETY: ptr is non null, and Pair / rascal_pair_t have the same layout
            return Ok(std::slice::from_raw_parts(ptr.cast(), count));
        }
    }
}

/// Convert a Simple System to a `rascal_system_t`
impl From<SimpleSystem> for rascal_system_t {
    fn from(system: SimpleSystem) -> rascal_system_t {
        unsafe extern fn size(this: *const c_void, size: *mut usize) {
            *size = (*this.cast::<SimpleSystem>()).size().unwrap();
        }

        unsafe extern fn species(this: *const c_void, species: *mut *const usize) {
            *species = (*this.cast::<SimpleSystem>()).species().unwrap().as_ptr();
        }

        unsafe extern fn positions(this: *const c_void, positions: *mut *const f64) {
            *positions = (*this.cast::<SimpleSystem>()).positions().unwrap().as_ptr().cast();
        }

        unsafe extern fn cell(this: *const c_void, cell: *mut f64) {
            let matrix = (*this.cast::<SimpleSystem>()).cell().unwrap().matrix();
            cell.add(0).write(matrix[0][0]);
            cell.add(1).write(matrix[0][1]);
            cell.add(2).write(matrix[0][2]);

            cell.add(3).write(matrix[1][0]);
            cell.add(4).write(matrix[1][1]);
            cell.add(5).write(matrix[1][2]);

            cell.add(6).write(matrix[2][0]);
            cell.add(7).write(matrix[2][1]);
            cell.add(8).write(matrix[2][2]);
        }

        unsafe extern fn compute_neighbors(this: *mut c_void, cutoff: f64) {
            (*this.cast::<SimpleSystem>()).compute_neighbors(cutoff).unwrap();
        }

        unsafe extern fn pairs(this: *const c_void, pairs: *mut *const rascal_pair_t, count: *mut usize) {
            let all_pairs = (*this.cast::<SimpleSystem>()).pairs().unwrap();
            *pairs = all_pairs.as_ptr().cast();
            *count = all_pairs.len();
        }

        unsafe extern fn pairs_containing(this: *const c_void, center: usize, pairs: *mut *const rascal_pair_t, count: *mut usize) {
            let all_pairs = (*this.cast::<SimpleSystem>()).pairs_containing(center).unwrap();
            *pairs = all_pairs.as_ptr().cast();
            *count = all_pairs.len();
        }

        rascal_system_t {
            user_data: Box::into_raw(Box::new(system)).cast(),
            size: Some(size),
            species: Some(species),
            positions: Some(positions),
            cell: Some(cell),
            compute_neighbors: Some(compute_neighbors),
            pairs: Some(pairs),
            pairs_containing: Some(pairs_containing),
        }
    }
}

/// Read all structures in the file at the given `path` using
/// [chemfiles](https://chemfiles.org/), and convert them to an array of
/// `rascal_system_t`.
///
/// This function can read all [formats supported by
/// chemfiles](https://chemfiles.org/chemfiles/latest/formats.html).
///
/// This function allocates memory, which must be released using
/// `rascal_basic_systems_free`.
///
/// If you need more control over the system behavior, consider writing your own
/// instance of `rascal_system_t`.
///
/// @param path path of the file to read from in the local filesystem
/// @param systems `*systems` will be set to a pointer to the first element of
///                 the array of `rascal_system_t`
/// @param count `*count` will be set to the number of systems read from the file
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::missing_panics_doc)]
pub unsafe extern fn rascal_basic_systems_read(
    path: *const c_char,
    systems: *mut *mut rascal_system_t,
    count: *mut usize,
) -> rascal_status_t {
    catch_unwind(move || {
        check_pointers!(path, systems, count);
        let path = CStr::from_ptr(path).to_str()?;
        let simple_systems = rascaline::systems::read_from_file(path)?;

        let mut c_systems = Vec::with_capacity(simple_systems.len());
        for system in simple_systems {
            c_systems.push(system.into());
        }

        // we rely on this below to drop the vector
        assert!(c_systems.capacity() == c_systems.len());

        *systems = c_systems.as_mut_ptr();
        *count = c_systems.len();
        std::mem::forget(c_systems);

        Ok(())
    })
}

/// Release memory allocated by `rascal_basic_systems_read`.
///
/// This function is only valid to call with a pointer to systems obtained from
/// `rascal_basic_systems_read`, and the corresponding `count`. Any other use
/// will probably result in segmentation faults or double free. If `systems` is
/// NULL, this function does nothing.
///
/// @param systems pointer to the first element of the array of
/// `rascal_system_t` @param count number of systems in the array
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_basic_systems_free(systems: *mut rascal_system_t, count: usize) -> rascal_status_t {
    catch_unwind(|| {
        if !systems.is_null() {
            let vec = Vec::from_raw_parts(systems, count, count);
            for element in vec {
                let boxed = Box::from_raw(element.user_data.cast::<SimpleSystem>());
                std::mem::drop(boxed);
            }
        }

        Ok(())
    })
}
