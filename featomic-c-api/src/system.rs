use std::os::raw::c_void;

use featomic::types::{Vector3D, Matrix3};
use featomic::systems::{SimpleSystem, Pair, UnitCell};
use featomic::{Error, System};

use crate::FEATOMIC_SYSTEM_ERROR;

use super::{catch_unwind, featomic_status_t};

/// Pair of atoms coming from a neighbor list
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct featomic_pair_t {
    /// index of the first atom in the pair
    pub first: usize,
    /// index of the second atom in the pair
    pub second: usize,
    /// distance between the two atoms
    pub distance: f64,
    /// vector from the first atom to the second atom, accounting for periodic
    /// boundary conditions. This should be
    /// `position[second] - position[first] + H * cell_shift`
    /// where `H` is the cell matrix.
    pub vector: [f64; 3],
    /// How many cell shift where applied to the `second` atom to create this
    /// pair.
    pub cell_shift_indices: [i32; 3],
}

/// A `featomic_system_t` deals with the storage of atoms and related information,
/// as well as the computation of neighbor lists.
///
/// This struct contains a manual implementation of a virtual table, allowing to
/// implement the rust `System` trait in C and other languages. Speaking in Rust
/// terms, `user_data` contains a pointer (analog to `Box<Self>`) to the struct
/// implementing the `System` trait; and then there is one function pointers
/// (`Option<unsafe extern fn(XXX)>`) for each function in the `System` trait.
///
/// The `featomic_status_t` return value for the function is used to communicate
/// error messages. It should be 0/`FEATOMIC_SUCCESS` in case of success, any
/// non-zero value in case of error. The error will be propagated to the
/// top-level caller as a `FEATOMIC_SYSTEM_ERROR`
///
/// A new implementation of the System trait can then be created in any language
/// supporting a C API (meaning any language for our purposes); by correctly
/// setting `user_data` to the actual data storage, and setting all function
/// pointers to the correct functions. For an example of code doing this, see
/// the `SystemBase` class in the Python interface to featomic.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, function
/// taking `const` pointer parameters can be called from multiple threads at the
/// same time. The `featomic_system_t` itself might be moved from one thread to
/// another.

// Function pointers have type `Option<unsafe extern fn(XXX)>`, where `Option`
// ensure that the `impl System for featomic_system_t` is forced to deal with the
// function pointer potentially being NULL. `unsafe` is required since these
// function come from another language and are not checked by the Rust compiler.
// Finally `extern` defaults to `extern "C"`, setting the ABI of the function to
// the default C ABI on the current system.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct featomic_system_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    user_data: *mut c_void,
    /// This function should set `*size` to the number of atoms in this system
    size: Option<unsafe extern fn(user_data: *const c_void, size: *mut usize) -> featomic_status_t>,
    /// This function should set `*types` to a pointer to the first element of
    /// a contiguous array containing the atomic types of each atom in the
    /// system. Different atomic types should be identified with a different
    /// value. These values are usually the atomic number, but don't have to be.
    /// The array should contain `featomic_system_t::size()` elements.
    types: Option<unsafe extern fn(user_data: *const c_void, types: *mut *const i32) -> featomic_status_t>,
    /// This function should set `*positions` to a pointer to the first element
    /// of a contiguous array containing the atomic cartesian coordinates.
    /// `positions[0], positions[1], positions[2]` must contain the x, y, z
    /// cartesian coordinates of the first atom, and so on.
    positions: Option<unsafe extern fn(user_data: *const c_void, positions: *mut *const f64) -> featomic_status_t>,
    /// This function should write the unit cell matrix in `cell`, which have
    /// space for 9 values. The cell should be written in row major order, i.e.
    /// `ax ay az bx by bz cx cy cz`, where a/b/c are the unit cell vectors.
    cell: Option<unsafe extern fn(user_data: *const c_void, cell: *mut f64) -> featomic_status_t>,
    /// This function should compute the neighbor list with the given cutoff,
    /// and store it for later access using `pairs` or `pairs_containing`.
    compute_neighbors: Option<unsafe extern fn(user_data: *mut c_void, cutoff: f64) -> featomic_status_t>,
    /// This function should set `*pairs` to a pointer to the first element of a
    /// contiguous array containing all pairs in this system; and `*count` to
    /// the size of the array/the number of pairs.
    ///
    /// This list of pair should only contain each pair once (and not twice as
    /// `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
    /// contains pairs where the distance between atoms is actually bellow the
    /// cutoff passed in the last call to `compute_neighbors`. This function is
    /// only valid to call after a call to `compute_neighbors`.
    pairs: Option<unsafe extern fn(user_data: *const c_void, pairs: *mut *const featomic_pair_t, count: *mut usize) -> featomic_status_t>,
    /// This function should set `*pairs` to a pointer to the first element of a
    /// contiguous array containing all pairs in this system containing the atom
    /// with index `atom`; and `*count` to the size of the array/the number of
    /// pairs.
    ///
    /// The same restrictions on the list of pairs as `featomic_system_t::pairs`
    /// applies, with the additional condition that the pair `i-j` should be
    /// included both in the return of `pairs_containing(i)` and
    /// `pairs_containing(j)`.
    pairs_containing: Option<unsafe extern fn(user_data: *const c_void, atom: usize, pairs: *mut *const featomic_pair_t, count: *mut usize) -> featomic_status_t>,
}

unsafe impl Send for featomic_system_t {}
unsafe impl Sync for featomic_system_t {}

impl<'a> System for &'a mut featomic_system_t {
    fn size(&self) -> Result<usize, Error> {
        let function = self.size.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.size function is NULL".into(),
        })?;

        let mut value = 0;
        let status = unsafe {
            function(self.user_data, &mut value)
        };

        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.size failed".into(),
            });
        }

        return Ok(value);
    }

    fn types(&self) -> Result<&[i32], Error> {
        let function = self.types.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.types function is NULL".into(),
        })?;

        let mut ptr = std::ptr::null();
        let status = unsafe {
            function(self.user_data, &mut ptr)
        };

        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.types failed".into(),
            });
        }

        let size = self.size()?;
        if ptr.is_null() && size != 0 {
            return Err(Error::External {
                status: FEATOMIC_SYSTEM_ERROR,
                message: "featomic_system_t.types returned a NULL pointer with non zero size".into(),
            });
        }

        if size == 0 {
            return Ok(&[])
        } else {
            unsafe {
                return Ok(std::slice::from_raw_parts(ptr, self.size()?));
            }
        }
    }

    fn positions(&self) -> Result<&[Vector3D], Error> {
        let function = self.positions.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.positions function is NULL".into(),
        })?;

        let mut ptr = std::ptr::null();
        let status = unsafe {
            function(self.user_data, &mut ptr)
        };
        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.positions failed".into(),
            });
        }

        let size = self.size()?;
        if ptr.is_null() && size != 0 {
            return Err(Error::External {
                status: FEATOMIC_SYSTEM_ERROR,
                message: "featomic_system_t.positions returned a NULL pointer with non zero size".into(),
            });
        }

        if size == 0 {
            return Ok(&[])
        } else {
            unsafe {
                return Ok(std::slice::from_raw_parts(ptr.cast(), self.size()?));
            }
        }
    }

    fn cell(&self) -> Result<UnitCell, Error> {
        let function = self.cell.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.cell function is NULL".into(),
        })?;

        let mut value = [[0.0; 3]; 3];
        let status = unsafe {
            function(self.user_data, &mut value[0][0])
        };

        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.cell failed".into(),
            });
        }

        let matrix = Matrix3::from(value);
        if matrix == Matrix3::zero() {
            Ok(UnitCell::infinite())
        } else {
            Ok(UnitCell::from(matrix))
        }
    }

    fn compute_neighbors(&mut self, cutoff: f64) -> Result<(), Error> {
        let function = self.compute_neighbors.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.compute_neighbors function is NULL".into(),
        })?;

        let status = unsafe {
            function(self.user_data, cutoff)
        };
        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.compute_neighbors failed".into(),
            });
        }
        Ok(())
    }

    fn pairs(&self) -> Result<&[Pair], Error> {
        let function = self.pairs.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.pairs function is NULL".into(),
        })?;

        let mut ptr = std::ptr::null();
        let mut count = 0;
        let status = unsafe {
            function(self.user_data, &mut ptr, &mut count)
        };
        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.pairs failed".into(),
            });
        }

        if ptr.is_null() && count != 0 {
            return Err(Error::External {
                status: FEATOMIC_SYSTEM_ERROR,
                message: "featomic_system_t.pairs returned a NULL pointer with non zero size".into(),
            });
        }

        if count == 0 {
            return Ok(&[])
        } else {
            unsafe {
                // SAFETY: ptr is non null, and Pair / featomic_pair_t have the same layout
                return Ok(std::slice::from_raw_parts(ptr.cast(), count));
            }
        }
    }

    fn pairs_containing(&self, atom: usize) -> Result<&[Pair], Error> {
        let function = self.pairs_containing.ok_or_else(|| Error::External {
            status: FEATOMIC_SYSTEM_ERROR,
            message: "featomic_system_t.pairs_containing function is NULL".into(),
        })?;

        let mut ptr = std::ptr::null();
        let mut count = 0;
        let status = unsafe {
            function(self.user_data, atom, &mut ptr, &mut count)
        };

        if !status.is_success() {
            return Err(Error::External {
                status: status.as_i32(),
                message: "call to featomic_system_t.pairs_containing failed".into(),
            });
        }

        if ptr.is_null() && count != 0 {
            return Err(Error::External {
                status: FEATOMIC_SYSTEM_ERROR,
                message: "featomic_system_t.pairs_containing returned a NULL pointer with non zero size".into(),
            });
        }

        if count == 0 {
            return Ok(&[])
        } else {
            unsafe {
                // SAFETY: ptr is non null, and Pair / featomic_pair_t have the same layout
                return Ok(std::slice::from_raw_parts(ptr.cast(), count));
            }
        }
    }
}

/// Convert a Simple System to a `featomic_system_t`
impl From<SimpleSystem> for featomic_system_t {
    fn from(system: SimpleSystem) -> featomic_system_t {
        unsafe extern fn size(this: *const c_void, size: *mut usize) -> featomic_status_t {
            catch_unwind(|| {
                *size = (*this.cast::<SimpleSystem>()).size()?;
                Ok(())
            })
        }

        unsafe extern fn types(this: *const c_void, types: *mut *const i32) -> featomic_status_t {
            catch_unwind(|| {
                *types = (*this.cast::<SimpleSystem>()).types()?.as_ptr();
                Ok(())
            })
        }

        unsafe extern fn positions(this: *const c_void, positions: *mut *const f64) -> featomic_status_t {
            catch_unwind(|| {
                *positions = (*this.cast::<SimpleSystem>()).positions()?.as_ptr().cast();
                Ok(())
            })
        }

        unsafe extern fn cell(this: *const c_void, cell: *mut f64) -> featomic_status_t {
            catch_unwind(|| {
                let matrix = (*this.cast::<SimpleSystem>()).cell()?.matrix();
                cell.add(0).write(matrix[0][0]);
                cell.add(1).write(matrix[0][1]);
                cell.add(2).write(matrix[0][2]);

                cell.add(3).write(matrix[1][0]);
                cell.add(4).write(matrix[1][1]);
                cell.add(5).write(matrix[1][2]);

                cell.add(6).write(matrix[2][0]);
                cell.add(7).write(matrix[2][1]);
                cell.add(8).write(matrix[2][2]);

                Ok(())
            })
        }

        unsafe extern fn compute_neighbors(this: *mut c_void, cutoff: f64) -> featomic_status_t {
            catch_unwind(|| {
                (*this.cast::<SimpleSystem>()).compute_neighbors(cutoff)?;

                Ok(())
            })
        }

        unsafe extern fn pairs(
            this: *const c_void,
            pairs: *mut *const featomic_pair_t,
            count: *mut usize,
        ) -> featomic_status_t {
            catch_unwind(|| {
                let all_pairs = (*this.cast::<SimpleSystem>()).pairs()?;
                *pairs = all_pairs.as_ptr().cast();
                *count = all_pairs.len();

                Ok(())
            })
        }

        unsafe extern fn pairs_containing(
            this: *const c_void,
            atom: usize,
            pairs: *mut *const featomic_pair_t,
            count: *mut usize,
        ) -> featomic_status_t {
            catch_unwind(|| {
                let all_pairs = (*this.cast::<SimpleSystem>()).pairs_containing(atom)?;
                *pairs = all_pairs.as_ptr().cast();
                *count = all_pairs.len();

                Ok(())
            })
        }

        featomic_system_t {
            user_data: Box::into_raw(Box::new(system)).cast(),
            size: Some(size),
            types: Some(types),
            positions: Some(positions),
            cell: Some(cell),
            compute_neighbors: Some(compute_neighbors),
            pairs: Some(pairs),
            pairs_containing: Some(pairs_containing),
        }
    }
}
