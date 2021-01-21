#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::redundant_field_names)]
#![allow(clippy::missing_errors_doc)]

mod utils;
#[macro_use]
mod status;
pub use self::status::{catch_unwind, rascal_status_t};

mod system;
mod descriptor;
mod calculator;
