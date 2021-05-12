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
///
/// The value 0 (`RASCAL_SUCCESS`) is used to indicate successful operations.
/// Positive non-zero values are reserved for internal use in rascaline.
/// Negative values are reserved for use in user code, in particular to indicate
/// error coming from callbacks.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[must_use]
pub struct rascal_status_t(i32);

impl rascal_status_t {
    pub fn is_success(self) -> bool {
        self.0 == RASCAL_SUCCESS
    }

    pub fn as_i32(self) -> i32 {
        self.0
    }
}

/// Status code used when a function succeeded
pub const RASCAL_SUCCESS: i32 = 0;
/// Status code used when a function got an invalid parameter
pub const RASCAL_INVALID_PARAMETER_ERROR: i32 = 1;
/// Status code used when there was an error reading or writing JSON
pub const RASCAL_JSON_ERROR: i32 = 2;
/// Status code used when a string contains non-utf8 data
pub const RASCAL_UTF8_ERROR: i32 = 3;
/// Status code used for error related to reading files with chemfiles
pub const RASCAL_CHEMFILES_ERROR: i32 = 4;
/// Status code used for errors coming from the system implementation if we
/// don't have a more specific status
pub const RASCAL_SYSTEM_ERROR: i32 = 128;
/// Status code used when there was an internal error, i.e. there is a bug
/// inside rascaline
pub const RASCAL_INTERNAL_ERROR: i32 = 255;


impl From<Error> for rascal_status_t {
    #[allow(clippy::match_same_arms)]
    fn from(error: Error) -> rascal_status_t {
        LAST_ERROR_MESSAGE.with(|message| {
            *message.borrow_mut() = CString::new(format!("{}", error)).expect("error message contains a null byte");
        });
        match error {
            Error::InvalidParameter(_) => rascal_status_t(RASCAL_INVALID_PARAMETER_ERROR),
            Error::Json(_) => rascal_status_t(RASCAL_JSON_ERROR),
            Error::Utf8(_) => rascal_status_t(RASCAL_UTF8_ERROR),
            Error::Chemfiles(_) => rascal_status_t(RASCAL_CHEMFILES_ERROR),
            Error::External{status, ..} => rascal_status_t(status),
            Error::Internal(_) => rascal_status_t(RASCAL_INTERNAL_ERROR),
            _ => rascal_status_t(RASCAL_INTERNAL_ERROR),
        }
    }
}

/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `rascal_status_t`.
pub fn catch_unwind<F>(function: F) -> rascal_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(_)) => rascal_status_t(RASCAL_SUCCESS),
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

    if status.0 != RASCAL_SUCCESS {
        eprintln!("ERROR: unable to get last error message!");
        return std::ptr::null();
    }

    return result;
}
