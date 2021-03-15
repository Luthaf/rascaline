#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names, clippy::upper_case_acronyms)]
#![allow(clippy::missing_errors_doc, clippy::missing_safety_doc, clippy::must_use_candidate)]

mod utils;
#[macro_use]
mod status;
pub use self::status::{catch_unwind, rascal_status_t};
pub use self::status::{RASCAL_SUCCESS, RASCAL_INVALID_PARAMETER_ERROR, RASCAL_JSON_ERROR};
pub use self::status::{RASCAL_UTF8_ERROR, RASCAL_CHEMFILES_ERROR, RASCAL_SYSTEM_ERROR};
pub use self::status::{RASCAL_INTERNAL_ERROR};

pub mod system;
pub mod descriptor;
pub mod calculator;

pub type LoggingCallback = Option<unsafe extern fn(message: *const std::os::raw::c_char)>;

static mut GLOBAL_CALLBACK: LoggingCallback = None;

#[no_mangle]
pub unsafe extern fn rascal_set_logging_callback(callback: LoggingCallback) {
    GLOBAL_CALLBACK = callback;
}


//use std::sync::Mutex;
//
//static mut GLOBAL_CALLBACK: Mutex<LoggingCallback> = Mutex::new(None);
//
//unsafe extern fn set_logging_callback(callback: LoggingCallback) {
//    let mut data = GLOBAL_CALLBACK.lock().unwrap();
//    *data  = callback;
//}
