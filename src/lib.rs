#![warn(clippy::all, clippy::pedantic)]

pub mod types;
pub use types::*;

pub mod system;
pub mod descriptor;
pub mod calculator;
pub mod c_api;
