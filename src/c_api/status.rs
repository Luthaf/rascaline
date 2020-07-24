use std::panic::UnwindSafe;
use std::cell::RefCell;

use crate::Error;

// Save the last error message in thread local storage.
//
// This is marginally better than a standare global static value because it
// allow multiple threads to each have separate errors conditions.
thread_local! {
    pub static LAST_ERROR_MESSAGE: RefCell<String> = RefCell::new("".into());
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
            *message.borrow_mut() = format!("{}", error);
        });
        match error {
            Error::InvalidParameter(_) => rascal_status_t::RASCAL_INVALID_PARAMETER_ERROR,
            Error::JSON(_) => rascal_status_t::RASCAL_JSON_ERROR,
            Error::Panic(_) => rascal_status_t::RASCAL_INTERNAL_PANIC,
        }
    }
}

pub fn catch_unwind<F>(function: F) -> rascal_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(_) => rascal_status_t::RASCAL_SUCCESS,
        Err(error) => Error::from(error).into()
    }
}

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
