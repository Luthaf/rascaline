

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Got an invalid parameter value in a function
    InvalidParameter(String),
    /// Error while serializing/deserializing data
    JSON(serde_json::Error),
    /// Error used when a panic was catched
    Panic(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidParameter(e) => write!(f, "invalid parameter: {}", e),
            Error::JSON(e) => write!(f, "json error: {}", e),
            Error::Panic(e) => write!(f, "internal error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InvalidParameter(_) => None,
            Error::JSON(e) => Some(e),
            Error::Panic(_) => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Error {
        Error::JSON(error)
    }
}


// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        if let Some(message) = error.downcast_ref::<String>() {
            Error::Panic(message.clone())
        } else {
            Error::Panic("panic message is not a string, something is very wrong".into())
        }
    }
}
