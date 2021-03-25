#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names, clippy::upper_case_acronyms)]
#![allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]

mod utils;
#[macro_use]
mod status;
pub use self::status::{catch_unwind, rascal_status_t};

pub mod system;
pub mod descriptor;
pub mod calculator;
