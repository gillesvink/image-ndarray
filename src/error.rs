use thiserror_no_std::Error;

/// Global error object for the image-ndarray crate.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[cfg(feature = "image")]
    #[error("NDArray had an error during initializaiton of shape: {0}")]
    NDArray(#[from] ndarray::ShapeError),
    #[error("Image could not be constructed from ndarray.")]
    ImageConstructFailed,
    #[error(
        "Image could not be constructed from ndarray because output does not match input channel count."
    )]
    ChannelMismatch,
}

pub type Result<T> = core::result::Result<T, Error>;
