#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names, clippy::upper_case_acronyms)]
#![allow(clippy::missing_errors_doc, clippy::missing_safety_doc, clippy::must_use_candidate)]

pub use log::{error, info, warn, Record, Level, Metadata, LevelFilter};
use std::ffi::{CString};

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


static mut GLOBAL_CALLBACK: RascalLoggingCallback = None;
struct RascalLogger;

#[no_mangle]
pub unsafe extern fn rascal_set_logging_callback(callback: RascalLoggingCallback) {
    GLOBAL_CALLBACK = callback;
    log::set_boxed_logger(Box::new(RascalLogger)).expect("Setting rascal logger faild.")
}


impl log::Log for RascalLogger {
    fn enabled(&self, _: &Metadata) -> bool {
       return true;
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = format!("{}:{} -- {}",
                record.level(),
                record.target(),
                record.args());
            let cstr = CString::new(message).unwrap();
            unsafe {
              GLOBAL_CALLBACK.unwrap()(record.level() as i32, cstr.as_ptr());
            }
        }
    }

    fn flush(&self) {}
}

//impl<T: ?Sized> Log for Box<T> {
//}

//use std::sync::Mutex;
//
//static mut GLOBAL_CALLBACK: Mutex<RascalLoggingCallback> = Mutex::new(None);
//
//unsafe extern fn set_logging_callback(callback: RascalLoggingCallback) {
//    let mut data = GLOBAL_CALLBACK.lock().unwrap();
//    *data  = callback;
//}
