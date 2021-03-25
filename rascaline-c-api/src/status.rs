use std::panic::UnwindSafe;
use std::cell::RefCell;
use std::os::raw::c_char;
use std::ffi::CString;

use rascaline::Error;

// Save the last error message in thread local storage.
//
// This is marginally better than a standard global static value because it
// allow multiple threads to each have separate errors conditions.
thread_local! {
    pub static LAST_ERROR_MESSAGE: RefCell<CString> = RefCell::new(CString::new("").expect("invalid C string"));
}

/// Status type returned by all functions in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum rascal_status_t {
    /// The function succeeded
    RASCAL_SUCCESS = 0,
    /// A function got an invalid parameter
    RASCAL_INVALID_PARAMETER_ERROR = 1,
    /// There was an error reading or writing JSON
    RASCAL_JSON_ERROR = 2,
    /// A string contains non-utf8 data
    RASCAL_UTF8_ERROR = 3,
    /// There was an error of unknown kind
    RASCAL_UNKNOWN_ERROR = 254,
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
            Error::Json(_) => rascal_status_t::RASCAL_JSON_ERROR,
            Error::Utf8(_) => rascal_status_t::RASCAL_UTF8_ERROR,
            Error::Panic(_) => rascal_status_t::RASCAL_INTERNAL_PANIC,
            _ => rascal_status_t::RASCAL_UNKNOWN_ERROR,
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
            return Err(rascaline::Error::InvalidParameter(
                format!("got invalid NULL pointer for {}", stringify!($pointer))
            ));
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers!($pointer);)*
    }
}

/// Get the last error message that was created on the current thread.
///
/// @returns the last error message, as a NULL-terminated string
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

    if status != rascal_status_t::RASCAL_SUCCESS {
        eprintln!("ERROR: unable to get last error message!");
        return std::ptr::null();
    }

    return result;
}
