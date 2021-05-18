#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names, clippy::upper_case_acronyms)]
#![allow(clippy::missing_errors_doc, clippy::missing_safety_doc, clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]

use std::ffi::{CString};
use std::sync::Mutex;

use log::{warn, Record, Metadata};
use lazy_static::lazy_static;

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

#[allow(non_camel_case_types)]
pub type rascal_logging_callback_t = Option<unsafe extern fn(level: i32, message: *const std::os::raw::c_char)>;

// Mutex cannot use rust static
// see https://stackoverflow.com/a/27826181
lazy_static! {
    static ref GLOBAL_CALLBACK: Mutex<rascal_logging_callback_t> = Mutex::new(None);
}

/// Implementation of `log::Log` that forward all log messages to the global
/// `rascal_logging_callback_t`.
struct RascalLogger;

#[no_mangle]
pub unsafe extern fn rascal_set_logging_callback(callback: rascal_logging_callback_t) {
    *GLOBAL_CALLBACK.lock().expect("mutex was poisoned") = callback;
    // we allow multiple sets of logger, therefore the result will be ignored
    let _ = log::set_boxed_logger(Box::new(RascalLogger));

    if cfg!(debug_assertions) {
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Info);
    }
}


impl log::Log for RascalLogger {
    fn enabled(&self, _: &Metadata) -> bool {
       return true;
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = format!("{} -- {}", record.target(), record.args());
            let message_cstr = CString::new(message).unwrap();
            unsafe {
                match *(GLOBAL_CALLBACK.lock().expect("mutex was poisoned")) {
                    Some(callback) => callback(record.level() as i32, message_cstr.as_ptr()),
                    None => unreachable!("missing callback but RascalLogger is set as the global logger"),
                }
            }
        }
    }

    fn flush(&self) {}
}
