use std::ops::{Deref, DerefMut};
use std::os::raw::c_char;

use crate::descriptor::Descriptor;

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
pub unsafe extern fn rascal_descriptor() -> *mut rascal_descriptor_t {
    let descriptor = Box::new(rascal_descriptor_t(Descriptor::new()));
    return Box::into_raw(descriptor);
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_free(descriptor: *mut rascal_descriptor_t) {
    if descriptor.is_null() {
        return;
    }
    Box::from_raw(descriptor);
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_values(descriptor: *const rascal_descriptor_t, data: *mut *const f64, environments: *mut usize, features: *mut usize) {
    assert!(!descriptor.is_null());
    assert!(!data.is_null());
    assert!(!environments.is_null());
    assert!(!features.is_null());

    let array = &(*descriptor).values;
    if array.is_empty() {
        *data = std::ptr::null();
    } else {
        *data = array.as_ptr();
    }

    let shape = array.shape();
    *environments = shape[0];
    *features = shape[1];
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_gradients(descriptor: *const rascal_descriptor_t, data: *mut *const f64, environments: *mut usize, features: *mut usize) {
    assert!(!descriptor.is_null());
    assert!(!data.is_null());
    assert!(!environments.is_null());
    assert!(!features.is_null());

    match &(*descriptor).gradient {
        Some(array) => {
            if array.is_empty() {
                *data = std::ptr::null();
            } else {
                *data = array.as_ptr();
            }

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
    size: *mut usize,
    count: *mut usize,
) {
    assert!(!descriptor.is_null());
    assert!(!values.is_null());
    assert!(!size.is_null());
    assert!(!count.is_null());

    let indexes = match indexes {
        rascal_indexes::RASCAL_INDEXES_FEATURES => &(*descriptor).features,
        rascal_indexes::RASCAL_INDEXES_ENVIRONMENTS => &(*descriptor).environments,
        rascal_indexes::RASCAL_INDEXES_GRADIENTS => {
            if let Some(indexes) = &(*descriptor).grad_envs {
                indexes
            } else {
                *values = std::ptr::null();
                *size = 0;
                *count = 0;
                return;
            }
        }
    };

    *size = indexes.size();
    *count = indexes.count();
    if *count != 0 {
        *values = &indexes.value(0)[0];
    } else {
        *values = std::ptr::null();
    }
}

#[no_mangle]
pub unsafe extern fn rascal_descriptor_indexes_names(
    descriptor: *const rascal_descriptor_t,
    indexes: rascal_indexes,
    names: *mut *const c_char,
    size: usize
) {
    assert!(!descriptor.is_null());
    assert!(!names.is_null());
    assert!(size >= 1);

    let indexes = match indexes {
        rascal_indexes::RASCAL_INDEXES_FEATURES => &(*descriptor).features,
        rascal_indexes::RASCAL_INDEXES_ENVIRONMENTS => &(*descriptor).environments,
        rascal_indexes::RASCAL_INDEXES_GRADIENTS => {
            if let Some(indexes) = &(*descriptor).grad_envs {
                indexes
            } else {
                for i in 0..size {
                    names.add(i).write(std::ptr::null());
                }
                return;
            }
        }
    };

    for (i, name) in indexes.c_names().iter().enumerate() {
        if i >= size {
            return;
        }
        names.add(i).write(name.as_ptr());
    }

    if size > indexes.size() {
        for i in indexes.size()..size {
            names.add(i).write(std::ptr::null());
        }
    }
}
