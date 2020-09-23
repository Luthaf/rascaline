#![warn(clippy::all, clippy::pedantic)]

pub mod types;
pub use types::*;

mod errors;
pub use self::errors::Error;

pub mod system;
pub mod descriptor;
pub mod calculator;
pub mod c_api;

mod calculators;

pub use system::System;
pub use calculator::Calculator;
pub use descriptor::Descriptor;
