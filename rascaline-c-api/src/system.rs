use std::os::raw::c_void;

use rascaline::types::{Vector3D, Matrix3};
use rascaline::system::{System, Pair, UnitCell};

/// Pair of atoms coming from a neighbor list
#[repr(C)]
pub struct rascal_pair_t {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
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
    /// a contiguous array containing the atomic species. Each different atomic
    /// species should be identified with a different value. These values are
    /// usually the atomic number, but don't have to be.
    species: Option<unsafe extern fn(user_data: *const c_void, species: *mut *const usize)>,
    /// This function should set `*positions` to a pointer to the first element
    /// of a contiguous array containing the atomic cartesian coordinates.
    /// `positions[0], positions[1], positions[2]` must contain the x, y, z
    /// cartesian coordinates of the first atom, and so on.
    positions: Option<unsafe extern fn(user_data: *const c_void, positions: *mut *const f64)>,
    /// This function should write the unit cell matrix in `cell`, which have
    /// space for 9 values.
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
    fn size(&self) -> usize {
        let mut value = 0;
        let function = self.size.expect("rascal_system_t.size is NULL");
        unsafe {
            function(self.user_data, &mut value);
        }
        return value;
    }

    fn species(&self) -> &[usize] {
        let mut ptr = std::ptr::null();
        let function = self.species.expect("rascal_system_t.species is NULL");
        unsafe {
            function(self.user_data, &mut ptr);
            // TODO: check if ptr.is_null() and error in some way?
            return std::slice::from_raw_parts(ptr, self.size());
        }
    }

    fn positions(&self) -> &[Vector3D] {
        let mut ptr = std::ptr::null();
        let function = self.positions.expect("rascal_system_t.positions is NULL");
        unsafe {
            function(self.user_data, &mut ptr);
            // TODO: check if ptr.is_null() and error in some way?
            return std::slice::from_raw_parts(ptr.cast(), self.size());
        }
    }

    fn cell(&self) -> UnitCell {
        let mut value = [[0.0; 3]; 3];
        let function = self.cell.expect("rascal_system_t.cell is NULL");
        let matrix: Matrix3 = unsafe {
            function(self.user_data, &mut value[0][0]);
            std::mem::transmute(value)
        };

        if matrix == Matrix3::zero() {
            return UnitCell::infinite();
        }

        return UnitCell::from(matrix);
    }

    fn compute_neighbors(&mut self, cutoff: f64) {
        let function = self.compute_neighbors.expect("rascal_system_t.compute_neighbors is NULL");
        unsafe {
            function(self.user_data, cutoff);
        }
    }

    fn pairs(&self) -> &[Pair] {
        let function = self.pairs.expect("rascal_system_t.pairs is NULL");
        let mut ptr = std::ptr::null();
        let mut count = 0;
        unsafe {
            function(self.user_data, &mut ptr, &mut count);
            return std::slice::from_raw_parts(ptr.cast(), count);
        }
    }

    fn pairs_containing(&self, center: usize) -> &[Pair] {
        let function = self.pairs_containing.expect("rascal_system_t.pairs_containing is NULL");
        let mut ptr = std::ptr::null();
        let mut count = 0;
        unsafe {
            function(self.user_data, center, &mut ptr, &mut count);
            return std::slice::from_raw_parts(ptr.cast(), count);
        }
    }
}
