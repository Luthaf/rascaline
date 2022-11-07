/// Euler's constant
pub const EULER: f64 = 0.57721566490153286;

mod gamma;
pub use self::gamma::{gamma, ln_gamma, gamma_ui, gamma_ur, gamma_lr, digamma};

mod hyp1f1;
pub(crate) use self::hyp1f1::hyp1f1;

mod hyp2f1;
pub use self::hyp2f1::hyp2f1;

mod double_regularized_1f1;
pub(crate) use self::double_regularized_1f1::DoubleRegularized1F1;

mod eigen;
pub(crate) use self::eigen::SymmetricEigen;

mod splines;
pub use self::splines::{HermitCubicSpline, SplineParameters};

mod spherical_harmonics;
pub use self::spherical_harmonics::{SphericalHarmonics, SphericalHarmonicsArray};
pub use self::spherical_harmonics::CachedAllocationsSphericalHarmonics;

mod k_vectors;
pub(crate) use self::k_vectors::compute_k_vectors;
pub(crate) use self::k_vectors::KVector;

mod cutoff;
pub(crate) use self::cutoff::{CutoffFunction, RadialScaling};
