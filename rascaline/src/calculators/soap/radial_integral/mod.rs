use ndarray::{ArrayViewMut2, Array2};

use crate::Error;
use crate::calculators::radial_basis::RadialBasis;

/// A `SoapRadialIntegral` computes the SOAP radial integral on a given radial
/// basis.
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// mor information on the radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
pub trait SoapRadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the radial integral for a single `distance` between two atoms
    /// and store the resulting data in the `(max_angular + 1) x max_radial`
    /// array `values`. If `gradients` is `Some`, also compute and store
    /// gradients there.
    fn compute(&self, rij: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>);
}

mod gto;
pub use self::gto::{SoapRadialIntegralGto, SoapRadialIntegralGtoParameters};

mod spline;
pub use self::spline::{SoapRadialIntegralSpline, SoapRadialIntegralSplineParameters};

/// Parameters controlling the radial integral for SOAP
#[derive(Debug, Clone, Copy)]
pub struct SoapRadialIntegralParameters {
    pub max_radial: usize,
    pub max_angular: usize,
    pub atomic_gaussian_width: f64,
    pub cutoff: f64,
}

/// Store together a Radial integral implementation and cached allocation for
/// values/gradients.
pub struct SoapRadialIntegralCache {
    /// Implementation of the radial integral
    code: Box<dyn SoapRadialIntegral>,
    /// Cache for the radial integral values
    pub(crate) values: Array2<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array2<f64>,
}

impl SoapRadialIntegralCache {
    /// Create a new `RadialIntegralCache` for the given radial basis & parameters
    pub fn new(radial_basis: RadialBasis, parameters: SoapRadialIntegralParameters) -> Result<Self, Error> {
        let code = match radial_basis {
            RadialBasis::Gto {splined_radial_integral, spline_accuracy} => {
                let parameters = SoapRadialIntegralGtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                let gto = SoapRadialIntegralGto::new(parameters)?;

                if splined_radial_integral {
                    let parameters = SoapRadialIntegralSplineParameters {
                        max_radial: parameters.max_radial,
                        max_angular: parameters.max_angular,
                        cutoff: parameters.cutoff,
                    };

                    Box::new(SoapRadialIntegralSpline::with_accuracy(
                        parameters, spline_accuracy, gto
                    )?)
                } else {
                    Box::new(gto) as Box<dyn SoapRadialIntegral>
                }
            }

            RadialBasis::TabulatedRadialIntegral {points} => {
                let parameters = SoapRadialIntegralSplineParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    cutoff: parameters.cutoff,
                };
                Box::new(SoapRadialIntegralSpline::from_tabulated(
                    parameters, points
                )?)
            }
        };

        let shape = (parameters.max_angular + 1, parameters.max_radial);
        let values = Array2::from_elem(shape, 0.0);
        let gradients = Array2::from_elem(shape, 0.0);

        return Ok(SoapRadialIntegralCache { code, values, gradients });
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
