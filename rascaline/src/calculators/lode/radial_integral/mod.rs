use ndarray::{ArrayViewMut2, Array1, Array2};

use crate::Error;
use crate::calculators::radial_basis::RadialBasis;

/// A `LodeRadialIntegral` computes the LODE radial integral on a given radial basis.
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// mor information on the radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
pub trait LodeRadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the LODE radial integral for a single k-vector `norm` and store
    /// the resulting data in the `(max_angular + 1) x max_radial` array
    /// `values`. If `gradients` is `Some`, also compute and store gradients
    /// there.
    fn compute(&self, k_norm: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);

    /// Compute the contribution of the central atom to the final `<n l m>`
    /// coefficients. By symmetry, only l=0 is non-zero, so this function
    /// returns a 1-D array containing the different `<n 0 0>` coefficients.
    ///
    /// This function differs from the rest of LODE calculation because it goes
    /// straight from atom => n l m, without using k-space projection in the
    /// middle.
    fn compute_center_contribution(&self) -> Array1<f64>;
}

mod gto;
pub use self::gto::{LodeRadialIntegralGto, LodeRadialIntegralGtoParameters};

mod spline;
pub use self::spline::{LodeRadialIntegralSpline, LodeRadialIntegralSplineParameters};

/// Parameters controlling the radial integral for LODE
#[derive(Debug, Clone, Copy)]
pub struct LodeRadialIntegralParameters {
    pub max_radial: usize,
    pub max_angular: usize,
    pub atomic_gaussian_width: f64,
    pub potential_exponent: usize,
    pub cutoff: f64,
    pub k_cutoff: f64,
}

/// Store together a Radial integral implementation and cached allocation for
/// values/gradients.
pub struct LodeRadialIntegralCache {
    /// Implementation of the radial integral
    code: Box<dyn LodeRadialIntegral>,
    /// Cache for the radial integral values
    pub(crate) values: Array2<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array2<f64>,

    /// Cache for the central atom contribution
    pub(crate) center_contribution: Array1<f64>,
}

impl LodeRadialIntegralCache {
    /// Create a new `RadialIntegralCache` for the given radial basis & parameters
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(radial_basis: RadialBasis, parameters: LodeRadialIntegralParameters) -> Result<Self, Error> {
        let code = match radial_basis {
            RadialBasis::Gto {splined_radial_integral, spline_accuracy} => {
                let gto_parameters = LodeRadialIntegralGtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    potential_exponent: parameters.potential_exponent,
                    cutoff: parameters.cutoff,
                };
                let gto = LodeRadialIntegralGto::new(gto_parameters)?;

                if splined_radial_integral {
                    let parameters = LodeRadialIntegralSplineParameters {
                        max_radial: parameters.max_radial,
                        max_angular: parameters.max_angular,
                        // the largest value the spline should interpolate is
                        // the k-space cutoff, not the real-space cutoff
                        // associated with the GTO basis
                        cutoff: parameters.k_cutoff,
                    };

                    Box::new(LodeRadialIntegralSpline::with_accuracy(
                        parameters, spline_accuracy, gto
                    )?)
                } else {
                    Box::new(gto) as Box<dyn LodeRadialIntegral>
                }
            }
            RadialBasis::TabulatedRadialIntegral {points, center_contribution} => {
                let parameters = LodeRadialIntegralSplineParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    cutoff: parameters.cutoff,
                };

                let center_contribution = center_contribution.ok_or(Error::InvalidParameter(
                    "For a tabulated radial integral with LODE please provide the
                    `center_contribution`.".into()))?;

                Box::new(LodeRadialIntegralSpline::from_tabulated(
                    parameters, points, center_contribution
                )?)
            }
        };
        let shape = (parameters.max_angular + 1, parameters.max_radial);
        let values = Array2::from_elem(shape, 0.0);
        let gradients = Array2::from_elem(shape, 0.0);
        let center_contribution = Array1::from_elem(parameters.max_radial, 0.0);

        return Ok(LodeRadialIntegralCache { code, values, gradients, center_contribution });
    }

    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub fn compute(&mut self, k_norm: f64, gradients: bool) {
        if gradients {
            self.code.compute(
                k_norm,
                self.values.view_mut(),
                Some(self.gradients.view_mut()),
            );
        } else {
            self.code.compute(
                k_norm,
                self.values.view_mut(),
                None,
            );
        }
    }

    /// Run `compute_center_contribution`, and store the results in
    /// `self.center_contributions`
    pub fn compute_center_contribution(&mut self) {
        self.center_contribution = self.code.compute_center_contribution();
    }
}
