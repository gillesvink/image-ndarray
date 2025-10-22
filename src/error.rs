use thiserror::Error;

/// Global error object for the image-ndarray crate.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("NDArray had an error during initializaiton of shape: {0}")]
    NDArray(#[from] ndarray::ShapeError),
    #[error("Image could not be constructed from ndarray.")]
    ImageConstructFailed,
    #[error("Image could not be constructed from ndarray because output does not match input channel count.")]
    ChannelMismatch
}

pub type Result<T> = std::result::Result<T, Error>;
