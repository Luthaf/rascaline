use std::str::Utf8Error;

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Got an invalid parameter value in a function
    InvalidParameter(String),
    /// Error while serializing/deserializing data
    Json(serde_json::Error),
    /// Error due to C strings containing non-utf8 data
    Utf8(Utf8Error),
    /// Error related to reading structure files
    Chemfiles(String),
    /// Error used when a panic was caught
    Panic(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidParameter(e) => write!(f, "invalid parameter: {}", e),
            Error::Json(e) => write!(f, "json error: {}", e),
            Error::Utf8(e) => write!(f, "utf8 decoding error: {}", e),
            Error::Chemfiles(e) => write!(f, "chemfiles error: {}", e),
            Error::Panic(e) => write!(f, "internal error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidParameter(_) |
            Error::Panic(_) |
            Error::Chemfiles(_) => None,
            Error::Json(e) => Some(e),
            Error::Utf8(e) => Some(e),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Error {
        Error::Json(error)
    }
}

impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Error {
        Error::Utf8(error)
    }
}


// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        let message = if let Some(message) = error.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = error.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            panic!("panic message is not a string, something is very wrong")
        };

        Error::Panic(message)
    }
}
