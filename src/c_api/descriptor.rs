use std::ops::{Deref, DerefMut};
use std::os::raw::c_char;
use std::ffi::CStr;

use crate::descriptor::Descriptor;
use super::{catch_unwind, rascal_status_t};

/// Opaque type representing a Descriptor
#[allow(non_camel_case_types)]
pub struct rascal_descriptor_t(Descriptor);

impl Deref for rascal_descriptor_t {
    type Target = Descriptor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for rascal_descriptor_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[no_mangle]
#[allow(clippy::module_name_repetitions)]
pub unsafe extern fn rascal_descriptor() -> *mut rascal_descriptor_t {
    let descriptor = Box::new(rascal_descriptor_t(Descriptor::new()));
    return Box::into_raw(descriptor);
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_free(descriptor: *mut rascal_descriptor_t) -> rascal_status_t {
    catch_unwind(|| {
        if !descriptor.is_null() {
            let boxed = Box::from_raw(descriptor);
            std::mem::drop(boxed);
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_values(
    descriptor: *const rascal_descriptor_t,
    data: *mut *const f64,
    environments: *mut usize,
    features: *mut usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, data, environments, features);

        let array = &(*descriptor).values;
        if array.is_empty() {
            *data = std::ptr::null();
        } else {
            *data = array.as_ptr();
        }

        let shape = array.shape();
        *environments = shape[0];
        *features = shape[1];

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_gradients(
    descriptor: *const rascal_descriptor_t,
    data: *mut *const f64,
    environments: *mut usize,
    features: *mut usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, data, environments, features);

        match &(*descriptor).gradients {
            Some(array) => {
                *data = array.as_ptr();
                let shape = array.shape();
                *environments = shape[0];
                *features = shape[1];
            }
            None => {
                *data = std::ptr::null();
                *environments = 0;
                *environments = 0;
            }
        }

        Ok(())
    })
}

#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum rascal_indexes {
    RASCAL_INDEXES_FEATURES = 0,
    RASCAL_INDEXES_ENVIRONMENTS = 1,
    RASCAL_INDEXES_GRADIENTS = 2,
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_indexes(
    descriptor: *const rascal_descriptor_t,
    indexes: rascal_indexes,
    values: *mut *const usize,
    count: *mut usize,
    size: *mut usize,
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, values, size, count);

        let indexes = match indexes {
            rascal_indexes::RASCAL_INDEXES_FEATURES => &(*descriptor).features,
            rascal_indexes::RASCAL_INDEXES_ENVIRONMENTS => &(*descriptor).environments,
            rascal_indexes::RASCAL_INDEXES_GRADIENTS => {
                if let Some(indexes) = &(*descriptor).gradients_indexes {
                    indexes
                } else {
                    *values = std::ptr::null();
                    *size = 0;
                    *count = 0;
                    return Ok(());
                }
            }
        };

        *size = indexes.size();
        *count = indexes.count();
        if *count == 0 {
            *values = std::ptr::null();
        } else {
            *values = &indexes[0][0];
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_indexes_names(
    descriptor: *const rascal_descriptor_t,
    indexes: rascal_indexes,
    names: *mut *const c_char,
    size: usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, names);

        let indexes = match indexes {
            rascal_indexes::RASCAL_INDEXES_FEATURES => &(*descriptor).features,
            rascal_indexes::RASCAL_INDEXES_ENVIRONMENTS => &(*descriptor).environments,
            rascal_indexes::RASCAL_INDEXES_GRADIENTS => {
                if let Some(indexes) = &(*descriptor).gradients_indexes {
                    indexes
                } else {
                    for i in 0..size {
                        names.add(i).write(std::ptr::null());
                    }
                    return Ok(());
                }
            }
        };

        for (i, name) in indexes.c_names().iter().enumerate() {
            if i >= size {
                // TODO: return an error instead if we don't have enough space
                // for all names?
                return Ok(());
            }
            names.add(i).write(name.as_ptr());
        }

        if size > indexes.size() {
            for i in indexes.size()..size {
                names.add(i).write(std::ptr::null());
            }
        }

        Ok(())
    })
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_densify(
    descriptor: *mut rascal_descriptor_t,
    variable: *const c_char,
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variable);
        let variable = CStr::from_ptr(variable).to_str()?;
        (*descriptor).densify(variable);
        Ok(())
    })
}
