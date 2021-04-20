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

pub type RascalLoggingCallback = Option<unsafe extern fn(level: i32, message: *const std::os::raw::c_char)>;

// Mutex cannot use rust static
// see https://stackoverflow.com/a/27826181
lazy_static! {
    static ref GLOBAL_CALLBACK: Mutex<RascalLoggingCallback> = Mutex::new(None);
}

struct RascalLogger;

#[no_mangle]
pub unsafe extern fn rascal_set_logging_callback(callback: RascalLoggingCallback, log_level: i32) {
    *GLOBAL_CALLBACK.lock().unwrap() = callback;
    // we allow multiple sets of logger, therefore the result will be ignored
    let _ = log::set_boxed_logger(Box::new(RascalLogger));
    // we could also only filter the messages on the python level, but I am
    // afraid this could result performance issues as the number of log messages
    // increases
    let log_level_enum = match log_level {
        0 => log::LevelFilter::Off,
        1 => log::LevelFilter::Error,
        2 => log::LevelFilter::Warn,
        3 => log::LevelFilter::Info,
        4 => log::LevelFilter::Debug,
        5 => log::LevelFilter::Trace,
        _ => panic!("Log level {} not known, please use an integer in the range [0, 5].", log_level)
    };
    let _ = log::set_max_level(log_level_enum);
}


impl log::Log for RascalLogger {
    fn enabled(&self, _: &Metadata) -> bool {
       return true;
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = format!("{} -- {}",
                record.target(),
                record.args());
            let message_cstr = CString::new(message).unwrap();
            unsafe {
                match *(GLOBAL_CALLBACK.lock().expect("Mutex was poisoned")) {
                    Some(callback) => callback(record.level() as i32, message_cstr.as_ptr()),
                    None => println!("No callback function was set."),
                }
            }
        }
    }

    fn flush(&self) {}
}
