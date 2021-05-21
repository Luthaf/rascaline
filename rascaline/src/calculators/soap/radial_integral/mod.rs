use ndarray::ArrayViewMut2;

pub trait RadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the radial integral for a single atomic `distance` and store the
    /// resulting data in the `max_radial x max_angular` array `values`. If
    /// `gradients` is `Some`, also compute and store gradients.
    fn compute(&self, rij: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);
}

mod hypergeometric;
pub use self::hypergeometric::{HyperGeometricSphericalExpansion, HyperGeometricParameters};

mod gto;
pub use self::gto::{GtoRadialIntegral, GtoParameters};

mod spline;
pub use self::spline::{SplinedRadialIntegral, SplinedRIParameters};
