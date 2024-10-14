use std::collections::BTreeMap;

use ndarray::{Array1, ArrayViewMut1};

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
    /// Implementations of the radial integral
    implementation: Box<dyn LodeRadialIntegral>,
    /// Cache for the radial integral values
    pub(crate) values: Array1<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array1<f64>,
}

impl LodeRadialIntegralCache {
    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub fn compute(&mut self, k_norm: f64, do_gradients: bool) {
        let gradient_view = if do_gradients {
            Some(self.gradients.view_mut())
        } else {
            None
        };

        self.implementation.compute(k_norm, self.values.view_mut(), gradient_view);
    }

    /// Get the size of this radial integral (i.e. number of radial basis)
    pub fn size(&self) -> usize {
        return self.implementation.size()
    }

    /// Get the size of this radial integral (i.e. number of radial basis)
    pub fn get_center_contribution(&self, density: DensityKind) -> Result<Array1<f64>, Error> {
        return self.implementation.get_center_contribution(density);
    }
}

/// Store all `LodeRadialIntegralCache` for different angular channels
pub struct LodeRadialIntegralCacheByAngular {
    pub(crate) by_angular: BTreeMap<usize, LodeRadialIntegralCache>,
}

impl LodeRadialIntegralCacheByAngular {
    /// Create a new `LodeRadialIntegralCacheByAngular` for the given radial basis & parameters
    pub fn new(
        density: DensityKind,
        basis: &SphericalExpansionBasis<LodeRadialBasis>,
        k_cutoff: f64
    ) -> Result<Self, Error> {
        match basis {
            SphericalExpansionBasis::TensorProduct(basis) => {
                let mut by_angular = BTreeMap::new();
                for o3_lambda in 0..=basis.max_angular {
                    // We only support some specific radial basis
                    let implementation = match basis.radial {
                        LodeRadialBasis::Gto { .. } => {
                            let gto = LodeRadialIntegralGto::new(&basis.radial, o3_lambda)?;

                            if let Some(accuracy) = basis.spline_accuracy {
                                let do_center_contribution = o3_lambda == 0;
                                Box::new(LodeRadialIntegralSpline::with_accuracy(
                                    gto, density, k_cutoff, accuracy, do_center_contribution
                                )?)
                            } else {
                                Box::new(gto) as Box<dyn LodeRadialIntegral>
                            }
                        },
                        LodeRadialBasis::Tabulated(ref tabulated) => {
                            Box::new(LodeRadialIntegralSpline::from_tabulated(
                                tabulated.clone(),
                                density,
                            )) as Box<dyn LodeRadialIntegral>
                        }
                    };

                    let size = implementation.size();
                    let values = Array1::from_elem(size, 0.0);
                    let gradients = Array1::from_elem(size, 0.0);

                    by_angular.insert(o3_lambda, LodeRadialIntegralCache {
                        implementation,
                        values,
                        gradients,
                    });
                }

                return Ok(LodeRadialIntegralCacheByAngular {
                    by_angular
                });
            }
        }
    }

    /// Run the calculation, the results are accessible with `get`
    pub fn compute(&mut self, distance: f64, do_gradients: bool) {
        self.by_angular.iter_mut().for_each(|(_, cache)| cache.compute(distance, do_gradients));
    }

    /// Get one of the individual cache, corresponding to the `o3_lambda`
    /// angular channel
    pub fn get(&self, o3_lambda: usize) -> Option<&LodeRadialIntegralCache> {
        self.by_angular.get(&o3_lambda)
    }

    pub(crate) fn get_mut(&mut self, o3_lambda: usize) -> Option<&mut LodeRadialIntegralCache> {
        self.by_angular.get_mut(&o3_lambda)
    }
}
