use ndarray::{Array1, Array2, ArrayViewMut1, Axis};

use crate::Error;
use crate::calculators::shared::{DensityKind, LodeRadialBasis, SphericalExpansionBasis};

/// A `LodeRadialIntegral` computes the LODE radial integral  for all radial
/// basis functions and a single spherical harmonic `l` channel
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// mor information on the radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
pub trait LodeRadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the LODE radial integral for a single k-vector `norm` and store
    /// the resulting data in the `values` array. If `gradients` is `Some`, also
    /// compute and store gradients there.
    fn compute(&self, k_norm: f64, values: ArrayViewMut1<f64>, gradients: Option<ArrayViewMut1<f64>>);

    /// Get how many basis functions are part of this integral. This is the
    /// shape to use for the `values` and `gradients` parameters to `compute`.
    fn size(&self) -> usize;

    /// Compute the contribution of the central atom to the final `<n l m>`
    /// coefficients. By symmetry, only l=0 is non-zero, so this function
    /// returns a 1-D array containing the different `<n 0 0>` coefficients.
    ///
    /// This function differs from the rest of LODE calculation because it goes
    /// straight from atom => n l m, without using k-space projection in the
    /// middle.
    fn get_center_contribution(&self, density: DensityKind) -> Result<Array1<f64>, Error>;
}

mod gto;
pub use self::gto::LodeRadialIntegralGto;

mod spline;
pub use self::spline::LodeRadialIntegralSpline;

/// Store together a Radial integral implementation and cached allocation for
/// values/gradients.
pub struct LodeRadialIntegralCache {
    /// Maximal value for the spherical harmonics angular moment
    max_angular: usize,
    /// Implementations of the radial integrals for each `l` in `0..angular_size`
    implementations: Vec<Box<dyn LodeRadialIntegral>>,
    /// Cache for the radial integral values
    pub(crate) values: Array2<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array2<f64>,

    /// Pre-computed central atom contribution
    pub(crate) center_contribution: Array1<f64>,
}

impl LodeRadialIntegralCache {
    /// Create a new `RadialIntegralCache` for the given radial basis & parameters
    pub fn new(density: DensityKind, basis: &SphericalExpansionBasis<LodeRadialBasis>, k_cutoff: f64) -> Result<Self, Error> {
        match basis {
            SphericalExpansionBasis::TensorProduct(basis) => {
                let mut implementations = Vec::new();
                let mut radial_size = 0;

                for l in 0..=basis.max_angular {
                    // We only support some specific radial basis
                    let implementation = match &basis.radial {
                        &LodeRadialBasis::Gto { .. } => {
                            let gto = LodeRadialIntegralGto::new(&basis.radial, l)?;

                            if let Some(accuracy) = basis.spline_accuracy {
                                Box::new(LodeRadialIntegralSpline::with_accuracy(
                                    gto, density, k_cutoff, accuracy
                                )?)
                            } else {
                                Box::new(gto) as Box<dyn LodeRadialIntegral>
                            }
                        },
                        LodeRadialBasis::Tabulated(tabulated) => {
                            Box::new(LodeRadialIntegralSpline::from_tabulated(
                                tabulated.clone(),
                                density,
                            )) as Box<dyn LodeRadialIntegral>
                        }
                    };

                    radial_size = implementation.size();
                    implementations.push(implementation);
                }

                let shape = [basis.max_angular + 1, radial_size];
                let values = Array2::from_elem(shape, 0.0);
                let gradients = Array2::from_elem(shape, 0.0);

                // the center contribution should use the same implementation
                // as the lambda=0 "radial" integral
                let center_contribution = implementations[0].get_center_contribution(density)?;

                return Ok(LodeRadialIntegralCache {
                    max_angular: basis.max_angular,
                    implementations,
                    values,
                    gradients,
                    center_contribution,
                });
            }
        }
    }

    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub fn compute(&mut self, k_norm: f64, gradients: bool) {
        if gradients {
            for l in 0..=self.max_angular {
                self.implementations[l].compute(
                    k_norm,
                    self.values.index_axis_mut(Axis(0), l),
                    Some(self.gradients.index_axis_mut(Axis(0), l)),
                );
            }
        } else {
            for l in 0..=self.max_angular {
                self.implementations[l].compute(
                    k_norm,
                    self.values.index_axis_mut(Axis(0), l),
                    None,
                );
            }
        }
    }

    /// Get the number of radial basis function for the radial integral
    /// associated with a given `o3_lambda`
    pub fn radial_size(&self, o3_lambda: usize) -> usize {
        self.implementations[o3_lambda].size()
    }
}
