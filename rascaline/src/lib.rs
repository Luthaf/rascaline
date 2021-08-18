#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::range_plus_one)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::module_name_repetitions)]

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap, clippy::cast_lossless, clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]

// Tests lints
#![cfg_attr(test, allow(clippy::float_cmp))]

pub mod types;
pub use types::*;

pub(crate) mod math;

mod errors;
pub use self::errors::Error;

pub mod systems;
pub use systems::{System, SimpleSystem};

pub mod descriptor;
pub use descriptor::Descriptor;

mod calculator;
pub use calculator::{Calculator, CalculationOptions, SelectedIndexes};

pub mod calculators;
