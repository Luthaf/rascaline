#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names, clippy::upper_case_acronyms)]
#![allow(clippy::missing_errors_doc, clippy::missing_safety_doc, clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]

mod utils;
#[macro_use]
mod status;
pub use self::status::{catch_unwind, rascal_status_t};
pub use self::status::{RASCAL_SUCCESS, RASCAL_INVALID_PARAMETER_ERROR, RASCAL_JSON_ERROR};
pub use self::status::{RASCAL_UTF8_ERROR, RASCAL_CHEMFILES_ERROR, RASCAL_SYSTEM_ERROR};
pub use self::status::{RASCAL_BUFFER_SIZE_ERROR, RASCAL_INTERNAL_ERROR};

mod logging;
pub use self::logging::{RASCAL_LOG_LEVEL_ERROR, RASCAL_LOG_LEVEL_WARN, RASCAL_LOG_LEVEL_INFO};
pub use self::logging::{RASCAL_LOG_LEVEL_DEBUG, RASCAL_LOG_LEVEL_TRACE};
pub use self::logging::{rascal_logging_callback_t, rascal_set_logging_callback};

pub mod system;
pub mod descriptor;
pub mod calculator;

pub mod profiling;
