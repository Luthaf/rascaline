use ndarray::{Array2, ArrayViewMut2};

use super::SoapRadialIntegral;
use crate::math::{HermitCubicSpline, SplineParameters, HermitSplinePoint};
use crate::calculators::radial_basis::SplinePoint;
use crate::Error;

/// `SoapRadialIntegralSpline` allows to evaluate another radial integral
/// implementation using [cubic Hermit spline][splines-wiki].
///
/// This can be much faster than using the actual radial integral
/// implementation.
///
/// [splines-wiki]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
pub struct SoapRadialIntegralSpline {
    spline: HermitCubicSpline<ndarray::Ix2>,
}

/// Parameters for computing the radial integral using Hermit cubic splines
#[derive(Debug, Clone, Copy)]
pub struct SoapRadialIntegralSplineParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// cutoff radius, this is also the maximal value that can be interpolated
    pub cutoff: f64,
}

impl SoapRadialIntegralSpline {
    /// Create a new `SoapRadialIntegralSpline` taking values from the given
    /// `radial_integral`. Points are added to the spline until the requested
    /// accuracy is reached. We consider that the accuracy is reached when
    /// either the mean absolute error or the mean relative error gets below the
    /// `accuracy` threshold.
    #[time_graph::instrument(name = "SoapRadialIntegralSpline::with_accuracy")]
    pub fn with_accuracy(
        parameters: SoapRadialIntegralSplineParameters,
        accuracy: f64,
        radial_integral: impl SoapRadialIntegral
    ) -> Result<SoapRadialIntegralSpline, Error> {
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
                let mut derivatives = Array2::from_elem(shape_tuple, 0.0);
                radial_integral.compute(x, values.view_mut(), Some(derivatives.view_mut()));
                (values, derivatives)
            },
        )?;

        return Ok(SoapRadialIntegralSpline { spline });
    }

    pub fn from_tabulated(
        parameters: SoapRadialIntegralSplineParameters,
        spline_points: Vec<SplinePoint>
    ) -> Result<SoapRadialIntegralSpline, Error> {

        let spline_parameters = SplineParameters {
            start: 0.0,
            stop: parameters.cutoff,
            shape: vec![parameters.max_angular + 1, parameters.max_radial],
        };

        let mut new_spline_points = Vec::new();
        for spline_point in spline_points {
            new_spline_points.push(
                HermitSplinePoint{
                    position: spline_point.position,
                    values: spline_point.values.0.clone(),
                    derivatives: spline_point.derivatives.0.clone(),
                }
            );
        }

        let spline = HermitCubicSpline::new(spline_parameters, new_spline_points);
        return Ok(SoapRadialIntegralSpline{spline});
    }
}

impl SoapRadialIntegral for SoapRadialIntegralSpline {
    #[time_graph::instrument(name = "SplinedRadialIntegral::compute")]
    fn compute(&self, x: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>) {
        self.spline.compute(x, values, gradients);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array;
    use serde_json::{json, Value};

    use super::*;
    use super::super::{SoapRadialIntegralGto, SoapRadialIntegralGtoParameters};

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE
        // is enough
        let parameters = SoapRadialIntegralSplineParameters {
            max_radial: 15,
            max_angular: 10,
            cutoff: 12.0,
        };

        let gto = SoapRadialIntegralGto::new(SoapRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        // this test only check that this code runs without crashing
        SoapRadialIntegralSpline::with_accuracy(parameters, 1e-10, gto).unwrap();
    }

    #[test]
    fn finite_difference() {
        let max_radial = 8;
        let max_angular = 8;
        let parameters = SoapRadialIntegralSplineParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
        };

        let gto = SoapRadialIntegralGto::new(SoapRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        // even with very bad accuracy, we want the gradients of the spline to
        // match the values produces by the spline, and not necessarily the
        // actual GTO gradients.
        let spline = SoapRadialIntegralSpline::with_accuracy(parameters, 1e-2, gto).unwrap();

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
            epsilon=delta, max_relative=1e-6
        );
    }


    #[derive(serde::Serialize)]
    /// Helper struct for testing de- and serialization of spline points
    struct HelperSplinePoint<D: ndarray::Dimension> {
        /// Position of the point
        pub(crate) position: f64,
        /// Values of the function to interpolate at the position
        pub(crate) values: Array<f64, D>,
        /// Derivatives of the function to interpolate at the position
        pub(crate) derivatives: Array<f64, D>,
    }


    /// Check that the `with_accuracy` spline can be directly loaded into
    /// `from_tabulated` and that both give the same result.
    #[test]
    fn accuracy_tabulated() {
        let max_radial = 8;
        let max_angular = 8;
        let parameters = SoapRadialIntegralSplineParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
        };

        let gto = SoapRadialIntegralGto::new(SoapRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        let spline_accuracy: SoapRadialIntegralSpline = SoapRadialIntegralSpline::with_accuracy(parameters, 1e-2, gto).unwrap();

        let mut new_spline_points = Vec::new();
        for spline_point in &spline_accuracy.spline.points {
            new_spline_points.push(
                HelperSplinePoint{
                    position: spline_point.position,
                    values: spline_point.values.clone(),
                    derivatives: spline_point.derivatives.clone(),
                }
            );
        }

        // Serialize and Deserialize spline points
        let spline_str = serde_json::to_string(&new_spline_points).unwrap();
        let spline_points: Vec<SplinePoint> = serde_json::from_str(&spline_str).unwrap();

        let spline_tabulated = SoapRadialIntegralSpline::from_tabulated(parameters, spline_points).unwrap();

        let rij = 3.4;
        let shape = (max_angular + 1, max_radial);

        let mut values_accuracy = Array2::from_elem(shape, 0.0);
        let mut gradients_accuracy = Array2::from_elem(shape, 0.0);
        spline_accuracy.compute(rij, values_accuracy.view_mut(), Some(gradients_accuracy.view_mut()));

        let mut values_tabulated = Array2::from_elem(shape, 0.0);
        let mut gradients_tabulated = Array2::from_elem(shape, 0.0);
        spline_tabulated.compute(rij, values_tabulated.view_mut(), Some(gradients_tabulated.view_mut()));

        assert_relative_eq!(
            values_accuracy, values_tabulated,
            epsilon=1e-15, max_relative=1e-16
        );

        assert_relative_eq!(
            gradients_accuracy, gradients_tabulated,
            epsilon=1e-15, max_relative=1e-16
        );

    }
}
