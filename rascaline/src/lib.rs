#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::range_plus_one)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::module_name_repetitions)]
#![allow(clippy::manual_assert, clippy::return_self_not_must_use, clippy::match_like_matches_macro)]
#![allow(clippy::needless_range_loop)]

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap, clippy::cast_lossless, clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]

// Tests lints
#![cfg_attr(test, allow(clippy::float_cmp))]

pub mod types;
pub use types::*;

pub mod math;

mod errors;
pub use self::errors::Error;

pub mod systems;
pub use self::systems::{System, SimpleSystem};

pub mod labels;

mod calculator;
pub use self::calculator::{Calculator, CalculationOptions, LabelsSelection};

pub mod calculators;

// only try to build the tutorials in test mode
#[cfg(test)]
mod tutorials;
