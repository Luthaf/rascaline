#![allow(clippy::missing_safety_doc, clippy::doc_markdown)]

mod utils;
#[macro_use]
mod status;
pub use self::status::{catch_unwind, featomic_status_t};
pub use self::status::{FEATOMIC_SUCCESS, FEATOMIC_INVALID_PARAMETER_ERROR, FEATOMIC_JSON_ERROR};
pub use self::status::{FEATOMIC_UTF8_ERROR, FEATOMIC_SYSTEM_ERROR};
pub use self::status::{FEATOMIC_BUFFER_SIZE_ERROR, FEATOMIC_INTERNAL_ERROR};

mod logging;
pub use self::logging::{FEATOMIC_LOG_LEVEL_ERROR, FEATOMIC_LOG_LEVEL_WARN, FEATOMIC_LOG_LEVEL_INFO};
pub use self::logging::{FEATOMIC_LOG_LEVEL_DEBUG, FEATOMIC_LOG_LEVEL_TRACE};
pub use self::logging::{featomic_logging_callback_t, featomic_set_logging_callback};

pub mod system;
pub mod calculator;

pub mod profiling;
