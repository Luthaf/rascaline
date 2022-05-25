use ndarray::ArrayViewMut2;

/// A `RadialIntegral` computes the radial integral on a given radial basis.
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// mor information on the radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
pub trait RadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the radial integral for a single `distance` between two atoms
    /// and store the resulting data in the `(max_angular + 1) x max_radial`
    /// array `values`. If `gradients` is `Some`, also compute and store
    /// gradients there.
    fn compute(&self, rij: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);
}

mod hypergeometric;
pub use self::hypergeometric::{HyperGeometricSphericalExpansion, HyperGeometricParameters};

mod gto;
pub use self::gto::{GtoRadialIntegral, GtoParameters};

mod spline;
pub use self::spline::{SplinedRadialIntegral, SplinedRIParameters};
