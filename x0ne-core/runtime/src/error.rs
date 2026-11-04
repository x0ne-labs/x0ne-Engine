use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// A file, a header, or a config that does not have the expected form.
    Format(String),
    /// An arena misuse caught at run time (foreign slice, out-of-range write).
    Arena(String),
    /// A planned entry point whose implementation belongs to a later gate.
    NotImplemented(&'static str),
    /// The CUDA driver, cuBLAS, or a kernel launch failed, or no usable GPU.
    Gpu(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Format(m) => write!(f, "format: {m}"),
            Error::Arena(m) => write!(f, "arena: {m}"),
            Error::NotImplemented(what) => write!(f, "not implemented yet: {what}"),
            Error::Gpu(m) => write!(f, "gpu: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub(crate) fn format_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Format(msg.into()))
}
