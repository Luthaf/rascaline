use std::collections::BTreeMap;

use ndarray::{Array1, ArrayViewMut1};

use crate::calculators::shared::DensityKind;
use crate::calculators::shared::{SphericalExpansionBasis, SoapRadialBasis};
use crate::Error;

/// A `SoapRadialIntegral` computes the SOAP radial integral for all radial
/// basis functions and a single spherical harmonic `l` channel
///
/// See equations 5 to 8 of [this paper](https://doi.org/10.1063/5.0044689) for
/// mor information on the radial integral.
///
/// `std::panic::RefUnwindSafe` is a required super-trait to enable passing
/// radial integrals across the C API. `Send` is a required super-trait to
/// enable passing radial integrals between threads.
#[allow(clippy::doc_markdown)]
pub trait SoapRadialIntegral: std::panic::RefUnwindSafe + Send {
    /// Compute the radial integral for a single `distance` between two atoms
    /// and store the resulting data in the `values` array. If `gradients` is
    /// `Some`, also compute and store gradients there.
    ///
    /// The radial integral $I_{nl}$ is defined as "the non-spherical harmonics
    /// part of the spherical expansion". Depending on the atomic density,
    /// different expressions can be used.
    ///
    /// For a delta density, the radial integral is simply the radial basis
    /// function $R_{nl}$ evaluated at the pair distance:
    ///
    /// $$ I_{nl}(r_{ij}) = R_{nl}(r_{ij}) $$
    ///
    /// For a Gaussian atomic density with a width of $\sigma$, the radial
    /// integral reduces to:
    ///
    /// $$
    /// I_{nl}(r_{ij}) = \frac{4\pi}{(\pi \sigma^2)^{3/4}} e^{-\frac{r_{ij}^2}{2\sigma^2}}
    ///     \int_0^\infty \mathrm{d}r r^2 R_{nl}(r) e^{-\frac{r^2}{2\sigma^2}} i_l\left(\frac{rr_{ij}}{\sigma^2}\right)
    /// $$
    ///
    /// where $i_l$ is the modified spherical Bessel function of the first kind
    /// of order $l$.
    ///
    /// Finally, for an arbitrary spherically symmetric atomic density `g(r)`,
    /// the radial integral is
    ///
    /// $$
    /// I_{nl}(r_{ij}) = 2\pi \int_0^\infty \mathrm{d}r r^2 R_{nl}(r)
    ///     \int_{-1}^1 \mathrm{d}u P_l(u) g(\sqrt{r^2+r_{ij}^2-2rr_{ij}u})
    /// $$
    ///
    /// where $P_l$ is the l-th Legendre polynomial.
    fn compute(&self, distance: f64, values: ArrayViewMut1<f64>, gradients: Option<ArrayViewMut1<f64>>);

    /// Get how many basis functions are part of this integral. This is the
    /// shape to use for the `values` and `gradients` parameters to `compute`.
    fn size(&self) -> usize;
}

mod gto;
pub use self::gto::SoapRadialIntegralGto;

mod spline;
pub use self::spline::SoapRadialIntegralSpline;

/// Store together a radial integral implementation and cached allocation for
/// values/gradients.
pub struct SoapRadialIntegralCache {
    /// Implementation of the radial integral
    implementation: Box<dyn SoapRadialIntegral>,
    /// Cache for the radial integral values
    pub(crate) values: Array1<f64>,
    /// Cache for the radial integral gradient
    pub(crate) gradients: Array1<f64>,
}

impl SoapRadialIntegralCache {
    fn new(
        o3_lambda: usize,
        radial: &SoapRadialBasis,
        density: DensityKind,
        cutoff: f64,
        spline_accuracy: Option<f64>,
    ) -> Result<SoapRadialIntegralCache, Error> {
        // We only support some specific combinations of density and basis
        let implementation = match (density, radial) {
            // Gaussian density + GTO basis
            (DensityKind::Gaussian {..}, &SoapRadialBasis::Gto { .. }) => {
                let gto = SoapRadialIntegralGto::new(cutoff, density, radial, o3_lambda)?;

                if let Some(accuracy) = spline_accuracy {
                    Box::new(SoapRadialIntegralSpline::with_accuracy(
                        gto, cutoff, accuracy
                    )?)
                } else {
                    Box::new(gto) as Box<dyn SoapRadialIntegral>
                }
            },
            // Dirac density + tabulated basis (also used for
            // tabulated radial integral with a different density)
            (DensityKind::DiracDelta, SoapRadialBasis::Tabulated(tabulated)) => {
                Box::new(SoapRadialIntegralSpline::from_tabulated(
                    tabulated.clone()
                )) as Box<dyn SoapRadialIntegral>
            }
            // Everything else is an error
            _ => {
                return Err(Error::InvalidParameter(
                    "this combination of basis and density is not supported in SOAP".into()
                ))
            }
        };

        let size = implementation.size();
        let values = Array1::from_elem(size, 0.0);
        let gradients = Array1::from_elem(size, 0.0);

        return Ok(SoapRadialIntegralCache {
            implementation,
            values,
            gradients,
        });
    }

    /// Run the calculation, the results are stored inside `self.values` and
    /// `self.gradients`
    pub fn compute(&mut self, distance: f64, do_gradients: bool) {
        let gradient_view = if do_gradients {
            Some(self.gradients.view_mut())
        } else {
            None
        };

        self.implementation.compute(distance, self.values.view_mut(), gradient_view);
    }
}

/// Store all `SoapRadialIntegralCache` for different angular channels
pub struct SoapRadialIntegralCacheByAngular {
    pub(crate) by_angular: BTreeMap<usize, SoapRadialIntegralCache>,
}

impl SoapRadialIntegralCacheByAngular {
    /// Create a new `SoapRadialIntegralCacheByAngular` for the given radial basis & density
    pub fn new(
        cutoff: f64,
        density: DensityKind,
        basis: &SphericalExpansionBasis<SoapRadialBasis>
    ) -> Result<SoapRadialIntegralCacheByAngular, Error> {
        match basis {
            SphericalExpansionBasis::TensorProduct(basis) => {
                let mut by_angular = BTreeMap::new();
                for o3_lambda in 0..=basis.max_angular {
                    let cache = SoapRadialIntegralCache::new(
                        o3_lambda,
                        &basis.radial,
                        density,
                        cutoff,
                        basis.spline_accuracy
                    )?;
                    by_angular.insert(o3_lambda, cache);
                }

                return Ok(SoapRadialIntegralCacheByAngular { by_angular });
            }
            SphericalExpansionBasis::Explicit(basis) => {
                let mut by_angular = BTreeMap::new();
                for (&o3_lambda, radial) in &*basis.by_angular {
                    let cache = SoapRadialIntegralCache::new(
                        o3_lambda,
                        radial,
                        density,
                        cutoff,
                        basis.spline_accuracy
                    )?;
                    by_angular.insert(o3_lambda, cache);
                }
                return Ok(SoapRadialIntegralCacheByAngular {
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
    pub fn get(&self, o3_lambda: usize) -> Option<&SoapRadialIntegralCache> {
        self.by_angular.get(&o3_lambda)
    }

    pub(crate) fn get_mut(&mut self, o3_lambda: usize) -> Option<&mut SoapRadialIntegralCache> {
        self.by_angular.get_mut(&o3_lambda)
    }
}
