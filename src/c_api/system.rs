use std::os::raw::c_void;

use crate::types::{Vector3D, Matrix3};
use crate::system::{System, Pair, UnitCell};


#[repr(C)]
pub struct rascal_system_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers
    user_data: *mut c_void,
    size: Option<unsafe extern fn(user_data: *const c_void, size: *mut usize)>,
    species: Option<unsafe extern fn(user_data: *const c_void, species: *mut *const usize)>,
    positions: Option<unsafe extern fn(user_data: *const c_void, positions: *mut *const f64)>,
    cell: Option<unsafe extern fn(user_data: *const c_void, cell: *mut f64)>,
    compute_neighbors: Option<unsafe extern fn(user_data: *mut c_void, cutoff: f64)>,
    pairs: Option<unsafe extern fn(user_data: *const c_void, pairs: *mut *const Pair, count: *mut usize)>,
}

impl System for rascal_system_t {
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
            let slice = std::slice::from_raw_parts(ptr as *const [f64; 3], self.size());
            // transmuting from &[[f64; 3]] to &[Vector3D] is safe since
            // Vector3D is repr(transparent)
            return std::mem::transmute(slice);
        }
    }

    fn cell(&self) -> UnitCell {
        let mut value = [[0.0; 3]; 3];
        let function = self.cell.expect("rascal_system_t.cell is NULL");
        let matrix: Matrix3 = unsafe {
            function(self.user_data, &mut value[0][0]);
            std::mem::transmute(value)
        };
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
            return std::slice::from_raw_parts(ptr, count);
        }
    }
}
