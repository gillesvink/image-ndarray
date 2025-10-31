#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
mod error;
mod traits;

pub use error::Error;
pub mod prelude;
