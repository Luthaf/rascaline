use std::ffi::{CString};
use std::sync::Mutex;

use log::{Record, Metadata};
use lazy_static::lazy_static;

/// The "error" level designates very serious errors
pub const RASCAL_LOG_LEVEL_ERROR: i32 = 1;

/// The "warn" level designates hazardous situations
pub const RASCAL_LOG_LEVEL_WARN: i32 = 2;

/// The "info" level designates useful information
pub const RASCAL_LOG_LEVEL_INFO: i32 = 3;

/// The "debug" level designates lower priority information
///
/// By default, log messages at this level are disabled in release mode, and
/// enabled in debug mode.
pub const RASCAL_LOG_LEVEL_DEBUG: i32 = 4;

/// The "trace" level designates very low priority, often extremely verbose,
/// information.
///
/// By default, rascaline disable this level, you can enable it by editing the
/// code.
pub const RASCAL_LOG_LEVEL_TRACE: i32 = 5;


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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_levels() {
        assert_eq!(RASCAL_LOG_LEVEL_ERROR, log::Level::Error as i32);
        assert_eq!(RASCAL_LOG_LEVEL_WARN, log::Level::Warn as i32);
        assert_eq!(RASCAL_LOG_LEVEL_INFO, log::Level::Info as i32);
        assert_eq!(RASCAL_LOG_LEVEL_DEBUG, log::Level::Debug as i32);
        assert_eq!(RASCAL_LOG_LEVEL_TRACE, log::Level::Trace as i32);
    }
}
