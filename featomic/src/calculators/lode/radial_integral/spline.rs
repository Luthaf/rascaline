use std::sync::Arc;

use ndarray::{Array1, ArrayViewMut1};

use super::LodeRadialIntegral;
use crate::calculators::shared::basis::radial::LodeTabulated;
use crate::calculators::shared::DensityKind;
use crate::math::{HermitCubicSpline, SplineParameters};
use crate::Error;

/// `LodeRadialIntegralSpline` allows to evaluate another radial integral
/// implementation using [cubic Hermit spline][splines-wiki].
///
/// This can be much faster than using the actual radial integral
/// implementation.
///
/// [splines-wiki]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
pub struct LodeRadialIntegralSpline {
    spline: Arc<HermitCubicSpline<ndarray::Ix1>>,
    density: DensityKind,
    center_contribution: Option<Array1<f64>>,
}

impl LodeRadialIntegralSpline {
    /// Create a new `LodeRadialIntegralSpline` taking values from the given
    /// `radial_integral`. Points are added to the spline until the requested
    /// accuracy is reached. We consider that the accuracy is reached when
    /// either the mean absolute error or the mean relative error gets below the
    /// `accuracy` threshold.
    #[time_graph::instrument(name = "LodeRadialIntegralSpline::with_accuracy")]
    pub fn with_accuracy(
        radial_integral: impl LodeRadialIntegral,
        density: DensityKind,
        k_cutoff: f64,
        accuracy: f64,
        with_center_contribution: bool,
    ) -> Result<LodeRadialIntegralSpline, Error> {
        let size = radial_integral.size();
        let spline_parameters = SplineParameters {
            start: 0.0,
            stop: k_cutoff,
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

        let mut center_contribution = None;
        if with_center_contribution {
            center_contribution = Some(radial_integral.get_center_contribution(density)?);
        }

        return Ok(LodeRadialIntegralSpline {
            spline: Arc::new(spline),
            density: density,
            center_contribution: center_contribution,
        });
    }

    /// Create a new `LodeRadialIntegralSpline` with user-defined spline points.
    ///
    /// The density/`tabulated.center_contribution` are assumed to match each
    /// other
    pub fn from_tabulated(tabulated: LodeTabulated, density: DensityKind) -> LodeRadialIntegralSpline {
        return LodeRadialIntegralSpline {
            spline: tabulated.spline,
            density: density,
            center_contribution: tabulated.center_contribution,
        };
    }
}

impl LodeRadialIntegral for LodeRadialIntegralSpline {
    fn size(&self) -> usize {
        self.spline.points[0].values.shape()[0]
    }

    #[time_graph::instrument(name = "SplinedRadialIntegral::compute")]
    fn compute(&self, x: f64, values: ArrayViewMut1<f64>, gradients: Option<ArrayViewMut1<f64>>) {
        self.spline.compute(x, values, gradients);
    }

    fn get_center_contribution(&self, density: DensityKind) -> Result<Array1<f64>, Error> {
        if density != self.density {
            return Err(Error::InvalidParameter("mismatched atomic density in splined LODE radial integral".into()));
        }

        if self.center_contribution.is_none() {
            return Err(Error::InvalidParameter(
                "`center_contribution` must be defined for the Tabulated radial \
                basis used with the L=0 angular channel".into()
            ));
        }

        return Ok(self.center_contribution.clone().expect("just checked"));
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::calculators::LodeRadialBasis;

    use super::*;
    use super::super::LodeRadialIntegralGto;

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE is enough
        let basis = LodeRadialBasis::Gto { max_radial: 15, radius: 5.0 };
        let gto = LodeRadialIntegralGto::new(&basis, 3).unwrap();

        let accuracy = 5e-10;
        let k_cutoff = 10.0;
        let density = DensityKind::SmearedPowerLaw { smearing: 0.5, exponent: 1 };

        // this test only check that this code runs without crashing
        LodeRadialIntegralSpline::with_accuracy(gto, density, k_cutoff, accuracy, true).unwrap();
    }

    #[test]
    fn finite_difference() {
        let radial_size = 8;
        let basis = LodeRadialBasis::Gto { max_radial: (radial_size - 1), radius: 5.0 };
        let gto = LodeRadialIntegralGto::new(&basis, 3).unwrap();

        let accuracy = 1e-2;
        let k_cutoff = 10.0;
        let density = DensityKind::SmearedPowerLaw { smearing: 0.5, exponent: 1 };

        // even with very bad accuracy, we want the gradients of the spline to match the
        // values produces by the spline, and not necessarily the actual GTO gradients.
        let spline = LodeRadialIntegralSpline::with_accuracy(gto, density, k_cutoff, accuracy, false).unwrap();

        let rij = 3.4;
        let delta = 1e-9;

        let mut values = Array1::from_elem(radial_size, 0.0);
        let mut values_delta = Array1::from_elem(radial_size, 0.0);
        let mut gradients = Array1::from_elem(radial_size, 0.0);
        spline.compute(rij, values.view_mut(), Some(gradients.view_mut()));
        spline.compute(rij + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;
        assert_relative_eq!(
            finite_differences, gradients,
            epsilon=delta, max_relative=5e-6
        );
    }
}
