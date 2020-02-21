use std::os::raw::c_char;
use std::ffi::CStr;

use crate::calculator::Calculator;
use super::REGISTERED_CALCULATORS;

#[repr(C)]
pub struct rascal_calculator_t {
    handle: Box<dyn Calculator>,
}

impl rascal_calculator_t {
    fn new(calculator: Box<dyn Calculator>) -> rascal_calculator_t {
        rascal_calculator_t {
            handle: calculator,
        }
    }
}

#[no_mangle]
pub unsafe extern fn rascal_calculator(name: *const c_char, parameters: *const c_char) -> *mut rascal_calculator_t {
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

    let boxed = Box::new(rascal_calculator_t::new(calculator));
    return Box::into_raw(boxed);
}

#[no_mangle]
pub unsafe extern fn rascal_calculator_free(calculator: *mut rascal_calculator_t) {
    let boxed = Box::from_raw(calculator);
    std::mem::drop(boxed);
}

#[no_mangle]
pub unsafe extern fn rascal_calculator_name(calculator: *const rascal_calculator_t, name: *mut c_char, bufflen: usize) {
    if calculator.is_null() {
        return;
    }

    let calculator_name = (*calculator).handle.name();
    let size = std::cmp::min(calculator_name.len(), bufflen - 1);

    std::ptr::copy(calculator_name.as_ptr(), name as *mut u8, size);
    // NULL-terminate the string
    name.add(size).write(0);
}
