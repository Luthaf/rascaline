use std::os::raw::c_char;
use std::ffi::CStr;
use std::ops::{Deref, DerefMut};

use crate::Calculator;
use crate::System;

use super::REGISTERED_CALCULATORS;
use super::utils::copy_str_to_c;

use super::descriptor::rascal_descriptor_t;
use super::system::rascal_system_t;

#[repr(C)]
pub struct rascal_calculator_t(Box<dyn Calculator>);

impl Deref for rascal_calculator_t {
    type Target = dyn Calculator;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl DerefMut for rascal_calculator_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

#[no_mangle]
pub unsafe extern fn rascal_calculator(name: *const c_char, parameters: *const c_char) -> *mut rascal_calculator_t {
    assert!(!name.is_null());
    assert!(!parameters.is_null());
    let name = CStr::from_ptr(name);
    let creator = match REGISTERED_CALCULATORS.get(&*name.to_string_lossy()) {
        Some(creator) => creator,
        None => return std::ptr::null_mut(),
    };

    let parameters = CStr::from_ptr(parameters);
    let calculator = match creator(&*parameters.to_string_lossy()) {
        Ok(calculator) => calculator,
        Err(e) => {
            eprintln!("{}", e);
            return std::ptr::null_mut()
        },
    };

    let boxed = Box::new(rascal_calculator_t(calculator));
    return Box::into_raw(boxed);
}

#[no_mangle]
pub unsafe extern fn rascal_calculator_free(calculator: *mut rascal_calculator_t) {
    if calculator.is_null() {
        return;
    }
    let boxed = Box::from_raw(calculator);
    std::mem::drop(boxed);
}

#[no_mangle]
pub unsafe extern fn rascal_calculator_name(calculator: *const rascal_calculator_t, name: *mut c_char, bufflen: usize) {
    assert!(!calculator.is_null());
    assert!(!name.is_null());
    copy_str_to_c(&(*calculator).name(), name, bufflen);
}

#[no_mangle]
pub unsafe extern fn rascal_calculator_compute(calculator: *mut rascal_calculator_t, descriptor: *mut rascal_descriptor_t, systems: *mut rascal_system_t, count: usize) {
    assert!(!calculator.is_null());
    assert!(!descriptor.is_null());
    assert!(!systems.is_null());

    let systems = std::slice::from_raw_parts_mut(systems, count);
    let mut references = Vec::new();
    for system in systems {
        references.push(system as &mut dyn System);
    }

    (*calculator).compute(&mut references, &mut *descriptor);
}
