use ndarray::{ArrayViewMut2, Array2};

use crate::Error;

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

mod gto;
pub use self::gto::{GtoRadialIntegral, GtoParameters};

mod spline;
pub use self::spline::{SplinedRadialIntegral, SplinedRIParameters};


#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// Radial basis that can be used in the spherical expansion
pub enum RadialBasis {
    /// Use a radial basis similar to Gaussian-Type Orbitals.
    ///
    /// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
    /// = cutoff * \sqrt{n} / n_max`
    Gto {
        /// compute the radial integral using splines. This is much faster than
        /// the base GTO implementation.
        #[serde(default = "serde_default_splined_radial_integral")]
        splined_radial_integral: bool,
        /// Accuracy for the spline. The number of control points in the spline
        /// is automatically determined to ensure the average absolute error is
        /// close to the requested accuracy.
        #[serde(default = "serde_default_spline_accuracy")]
        spline_accuracy: f64,
    },
}

fn serde_default_splined_radial_integral() -> bool { true }
fn serde_default_spline_accuracy() -> f64 { 1e-8 }

/// Parameters controlling the radial basis for SOAP
#[derive(Debug, Clone, Copy)]
pub struct RadialBasisParameters {
    pub max_radial: usize,
    pub max_angular: usize,
    pub atomic_gaussian_width: f64,
    pub cutoff: f64,
}

impl RadialBasis {
    fn construct(&self, parameters: RadialBasisParameters) -> Result<Box<dyn RadialIntegral>, Error> {
        match self {
            RadialBasis::Gto {splined_radial_integral, spline_accuracy} => {
                let parameters = GtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                let gto = GtoRadialIntegral::new(parameters)?;

                if !splined_radial_integral {
                    return Ok(Box::new(gto));
                }

                let parameters = SplinedRIParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    cutoff: parameters.cutoff,
                };

                return Ok(Box::new(SplinedRadialIntegral::with_accuracy(
                    parameters, *spline_accuracy, gto
                )?));
            }
        };
    }
}

/// Store together a Radial integral implementation and cached allocation for
/// values/gradients.
pub struct RadialIntegralCache {
    /// Implementation of the radial integral
    code: Box<dyn RadialIntegral>,
    /// Cache for the radial integral values
    pub(crate) values: Array2<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array2<f64>,
}

impl RadialIntegralCache {
    /// Create a new `RadialIntegralCache` for the given radial basis & parameters
    pub fn new(radial_basis: &RadialBasis, parameters: RadialBasisParameters) -> Result<Self, Error> {
        let code = radial_basis.construct(parameters)?;
        let shape = (parameters.max_angular + 1, parameters.max_radial);
        let values = Array2::from_elem(shape, 0.0);
        let gradients = Array2::from_elem(shape, 0.0);

        return Ok(RadialIntegralCache { code, values, gradients });
    }

    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub fn compute(&mut self, distance: f64, gradients: bool) {
        if gradients {
            self.code.compute(
                distance,
                self.values.view_mut(),
                Some(self.gradients.view_mut()),
            );
        } else {
            self.code.compute(
                distance,
                self.values.view_mut(),
                None,
            );
        }
    }
}
