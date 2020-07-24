use std::panic::UnwindSafe;
use std::cell::RefCell;
use std::os::raw::c_char;
use std::ffi::CString;

use crate::Error;

// Save the last error message in thread local storage.
//
// This is marginally better than a standare global static value because it
// allow multiple threads to each have separate errors conditions.
thread_local! {
    pub static LAST_ERROR_MESSAGE: RefCell<CString> = RefCell::new(CString::new("").expect("invalid C string"));
}

/// Status type returned by all functions in the C API.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rascal_status_t {
    /// The function succeeded
    RASCAL_SUCCESS = 0,
    /// A function got an invalid parameter
    RASCAL_INVALID_PARAMETER_ERROR = 1,
    /// There was an error reading or writting JSON
    RASCAL_JSON_ERROR = 2,
    /// There was an internal error (rust panic)
    RASCAL_INTERNAL_PANIC = 255,
}

impl From<Error> for rascal_status_t {
    fn from(error: Error) -> rascal_status_t {
        LAST_ERROR_MESSAGE.with(|message| {
            *message.borrow_mut() = CString::new(format!("{}", error)).expect("error message contains a null byte");
        });
        match error {
            Error::InvalidParameter(_) => rascal_status_t::RASCAL_INVALID_PARAMETER_ERROR,
            Error::JSON(_) => rascal_status_t::RASCAL_JSON_ERROR,
            Error::Panic(_) => rascal_status_t::RASCAL_INTERNAL_PANIC,
        }
    }
}

/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `rascal_status_t`.
pub fn catch_unwind<F>(function: F) -> rascal_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(_)) => rascal_status_t::RASCAL_SUCCESS,
        Ok(Err(error)) => error.into(),
        Err(error) => Error::from(error).into()
    }
}

/// Check that pointers (used as C API function parameters) are not null.
#[macro_export]
macro_rules! check_pointers {
    ($pointer: ident) => {
        if $pointer.is_null() {
            return Err($crate::Error::InvalidParameter(
                format!("got invalid NULL pointer for {}", stringify!($pointer))
            ));
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers!($pointer);)*
    }
}

/// Get the last error message that was sent on the current thread
#[no_mangle]
pub unsafe extern fn rascal_last_error() -> *const c_char {
    let mut result = std::ptr::null();
    let wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        LAST_ERROR_MESSAGE.with(|message| {
            *wrapper.0 = message.borrow().as_ptr();
        });
        Ok(())
    });

    if status == rascal_status_t::RASCAL_SUCCESS {
        return result;
    } else {
        eprintln!("ERROR: unable to get last error message!");
        return std::ptr::null();
    }
}
