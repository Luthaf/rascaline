use ndarray::{Array1, Array2, ArrayViewMut2};

use super::LodeRadialIntegral;
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
    spline: HermitCubicSpline<ndarray::Ix2>,
    center_contribution: ndarray::Array1<f64>,
}

/// Parameters for computing the radial integral using Hermit cubic splines
#[derive(Debug, Clone, Copy)]
pub struct LodeRadialIntegralSplineParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// cutoff radius, this is also the maximal value that can be interpolated
    pub cutoff: f64,
}

impl LodeRadialIntegralSpline {
    /// Create a new `LodeRadialIntegralSpline` taking values from the given
    /// `radial_integral`. Points are added to the spline until the requested
    /// accuracy is reached. We consider that the accuracy is reached when
    /// either the mean absolute error or the mean relative error gets below the
    /// `accuracy` threshold.
    #[time_graph::instrument(name = "LodeRadialIntegralSpline::with_accuracy")]
    pub fn with_accuracy(
        parameters: LodeRadialIntegralSplineParameters,
        accuracy: f64,
        radial_integral: impl LodeRadialIntegral
    ) -> Result<LodeRadialIntegralSpline, Error> {
        let shape_tuple = (parameters.max_angular + 1, parameters.max_radial);

        let parameters = SplineParameters {
            start: 0.0,
            stop: parameters.cutoff,
            shape: vec![parameters.max_angular + 1, parameters.max_radial],
        };

        let spline = HermitCubicSpline::with_accuracy(
            accuracy,
            parameters,
            |x| {
                let mut values = Array2::from_elem(shape_tuple, 0.0);
                let mut gradients = Array2::from_elem(shape_tuple, 0.0);
                radial_integral.compute(x, values.view_mut(), Some(gradients.view_mut()));
                (values, gradients)
            },
        )?;

        return Ok(LodeRadialIntegralSpline {
            spline,
            center_contribution: radial_integral.compute_center_contribution()
        });
    }
}

impl LodeRadialIntegral for LodeRadialIntegralSpline {
    #[time_graph::instrument(name = "SplinedRadialIntegral::compute")]
    fn compute(&self, x: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>) {
        self.spline.compute(x, values, gradients);
    }

    fn compute_center_contribution(&self) -> Array1<f64> {
        return self.center_contribution.clone();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use super::super::{LodeRadialIntegralGto, LodeRadialIntegralGtoParameters};

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE
        // is enough
        let parameters = LodeRadialIntegralSplineParameters {
            max_radial: 15,
            max_angular: 10,
            cutoff: 12.0,
        };

        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        // this test only check that this code runs without crashing
        LodeRadialIntegralSpline::with_accuracy(parameters, 5e-10, gto).unwrap();
    }

    #[test]
    fn finite_difference() {
        let max_radial = 8;
        let max_angular = 8;
        let parameters = LodeRadialIntegralSplineParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
        };

        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        // even with very bad accuracy, we want the gradients of the spline to
        // match the values produces by the spline, and not necessarily the
        // actual GTO gradients.
        let spline = LodeRadialIntegralSpline::with_accuracy(parameters, 1e-2, gto).unwrap();

        let rij = 3.4;
        let delta = 1e-9;

        let shape = (max_angular + 1, max_radial);
        let mut values = Array2::from_elem(shape, 0.0);
        let mut values_delta = Array2::from_elem(shape, 0.0);
        let mut gradients = Array2::from_elem(shape, 0.0);
        spline.compute(rij, values.view_mut(), Some(gradients.view_mut()));
        spline.compute(rij + delta, values_delta.view_mut(), None);

        let finite_differences = (&values_delta - &values) / delta;
        assert_relative_eq!(
            finite_differences, gradients,
            epsilon=delta, max_relative=5e-6
        );
    }
}
