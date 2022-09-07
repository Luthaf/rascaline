use ndarray::ArrayViewMut2;

/// A `RadialIntegral` computes a radial integral on a given radial basis.
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// more information on the SOAP radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
pub trait RadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the radial integral for at `x` (`x` is the distance between
    /// atoms for SOAP, or the k-vector norm for Lode) and store the resulting
    /// data in the `(max_angular + 1) x max_radial` array `values`. If
    /// `gradients` is `Some`, also compute and store gradients there.
    fn compute(&self, x: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);
}

mod gto_radial_basis;
pub use self::gto_radial_basis::GtoRadialBasis;

mod spline;
pub use self::spline::{SplinedRadialIntegral, SplinedRIParameters};
