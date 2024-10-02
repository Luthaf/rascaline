use std::sync::Arc;

use ndarray::{Array1, ArrayViewMut1};

use super::SoapRadialIntegral;
use crate::calculators::shared::basis::radial::Tabulated;
use crate::math::{HermitCubicSpline, SplineParameters};
use crate::Error;

/// `SoapRadialIntegralSpline` allows to evaluate another radial integral
/// implementation using [cubic Hermit spline][splines-wiki].
///
/// This can be much faster than using analytical radial integral
/// implementations.
///
/// [splines-wiki]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
pub struct SoapRadialIntegralSpline {
    spline: Arc<HermitCubicSpline<ndarray::Ix1>>,
}

impl SoapRadialIntegralSpline {
    /// Create a new `SoapRadialIntegralSpline` taking values from the given
    /// `radial_integral`. Points are added to the spline between 0 and `cutoff`
    /// until the requested `accuracy` is reached. We consider that the accuracy
    /// is reached when either the mean absolute error or the mean relative
    /// error gets below the `accuracy` threshold.
    #[time_graph::instrument(name = "SoapRadialIntegralSpline::with_accuracy")]
    pub fn with_accuracy(
        radial_integral: impl SoapRadialIntegral,
        cutoff: f64,
        accuracy: f64,
    ) -> Result<SoapRadialIntegralSpline, Error> {
        let size = radial_integral.size();
        let spline_parameters = SplineParameters {
            start: 0.0,
            stop: cutoff,
            shape: vec![size],
        };

        let spline = HermitCubicSpline::with_accuracy(
            accuracy,
            spline_parameters,
            |x| {
                let mut values = Array1::from_elem(size, 0.0);
                let mut derivatives = Array1::from_elem(size, 0.0);
                radial_integral.compute(x, values.view_mut(), Some(derivatives.view_mut()));
                (values, derivatives)
            },
        )?;

        return Ok(SoapRadialIntegralSpline { spline: Arc::new(spline) });
    }

    /// Create a new `SoapRadialIntegralSpline` with user-defined spline points.
    pub fn from_tabulated(tabulated: Tabulated) -> SoapRadialIntegralSpline {
        return SoapRadialIntegralSpline {
            spline: tabulated.spline
        };
    }
}

impl SoapRadialIntegral for SoapRadialIntegralSpline {
    fn size(&self) -> usize {
        self.spline.points[0].values.shape()[0]
    }

    #[time_graph::instrument(name = "SplinedRadialIntegral::compute")]
    fn compute(&self, x: f64, values: ArrayViewMut1<f64>, gradients: Option<ArrayViewMut1<f64>>) {
        self.spline.compute(x, values, gradients);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::calculators::shared::{DensityKind, SoapRadialBasis};

    use super::*;
    use super::super::SoapRadialIntegralGto;

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE
        // is enough
        let density = DensityKind::Gaussian { width: 0.5 };
        let basis = SoapRadialBasis::Gto { max_radial: 15 };
        let gto_ri = SoapRadialIntegralGto::new(12.0, density, &basis, 0).unwrap();

        // this test only check that this code runs without crashing
        SoapRadialIntegralSpline::with_accuracy(gto_ri, 12.0, 1e-10).unwrap();
    }

    #[test]
    fn finite_difference() {
        let density = DensityKind::Gaussian { width: 0.5 };
        let basis = SoapRadialBasis::Gto { max_radial: 8 };
        let gto_ri = SoapRadialIntegralGto::new(5.0, density, &basis, 0).unwrap();

        // even with very bad accuracy, we want the gradients of the spline to
        // match the values produces by the spline, and not necessarily the
        // actual GTO gradients.
        let spline = SoapRadialIntegralSpline::with_accuracy(gto_ri, 5.0, 1e-2).unwrap();

        let x = 3.4;
        let delta = 1e-9;

        let size = spline.size();
        let mut values = Array1::from_elem(size, 0.0);
        let mut values_delta = Array1::from_elem(size, 0.0);
        let mut gradients = Array1::from_elem(size, 0.0);
        spline.compute(x, values.view_mut(), Some(gradients.view_mut()));
        spline.compute(x + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;
        assert_relative_eq!(
            finite_differences, gradients,
            epsilon=delta, max_relative=1e-6
        );
    }
}
