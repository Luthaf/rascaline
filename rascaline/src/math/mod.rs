/// Euler's constant
pub const EULER: f64 = 0.5772156649015329;

mod double_regularized_1f1;
pub(crate) use self::double_regularized_1f1::DoubleRegularized1F1;

mod eigen;
pub(crate) use self::eigen::SymmetricEigen;

mod exp;
pub use self::exp::{exp1, expi};

mod erf;
pub use self::erf::{erf, erfc};

mod gamma;
pub(crate) use self::gamma::{gamma, ln_gamma, digamma};

mod hyp1f1;
pub(crate) use self::hyp1f1::hyp1f1;

mod hyp2f1;
pub (crate) use self::hyp2f1::hyp2f1;

mod splines;
pub(crate) use self::splines::{HermitSplinePoint, HermitCubicSpline, SplineParameters};

mod spherical_harmonics;
pub use self::spherical_harmonics::{SphericalHarmonics, SphericalHarmonicsArray};
pub(crate) use self::spherical_harmonics::SphericalHarmonicsCache;

mod k_vectors;
pub use self::k_vectors::KVector;
pub use self::k_vectors::compute_k_vectors;
