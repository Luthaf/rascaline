use ndarray::{Array1, Array2, ArrayViewMut2};

use super::LodeRadialIntegral;
use crate::math::{HermitCubicSpline, SplineParameters, HermitSplinePoint};
use crate::calculators::radial_basis::SplinePoint;
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
    /// k-space cutoff radius, this is also the maximal value that can be interpolated
    pub k_cutoff: f64,
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
            stop: parameters.k_cutoff,
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

    /// Create a new `LodeRadialIntegralSpline` with user-defined spline points.
    pub fn from_tabulated(
        parameters: LodeRadialIntegralSplineParameters,
        spline_points: Vec<SplinePoint>,
        center_contribution: Vec<f64>,
    ) -> Result<LodeRadialIntegralSpline, Error> {

        let spline_parameters = SplineParameters {
            start: 0.0,
            stop: parameters.k_cutoff,
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

        if center_contribution.len() != parameters.max_radial {
            return Err(Error::InvalidParameter(format!(
                "wrong length of center_contribution, expected {} elements but got {}",
                parameters.max_radial, center_contribution.len()
            )))
        }

        let spline = HermitCubicSpline::new(spline_parameters, new_spline_points);
        return Ok(LodeRadialIntegralSpline{
            spline: spline, center_contribution: Array1::from_vec(center_contribution)});
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
    use ndarray::Array;

    use super::*;
    use super::super::{LodeRadialIntegralGto, LodeRadialIntegralGtoParameters};

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE is enough
        let parameters = LodeRadialIntegralSplineParameters {
            max_radial: 15,
            max_angular: 10,
            k_cutoff: 10.0,
        };

        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: 5.0,
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
            k_cutoff: 10.0,
        };

        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        // even with very bad accuracy, we want the gradients of the spline to match the
        // values produces by the spline, and not necessarily the actual GTO gradients.
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
        let parameters = LodeRadialIntegralSplineParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            k_cutoff: 10.0,
        };

        let gto = LodeRadialIntegralGto::new(LodeRadialIntegralGtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: 5.0,
            atomic_gaussian_width: 0.5,
            potential_exponent: 1,
        }).unwrap();

        let spline_accuracy: LodeRadialIntegralSpline = LodeRadialIntegralSpline::with_accuracy(parameters, 1e-2, gto).unwrap();

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

        let spline_tabulated = LodeRadialIntegralSpline::from_tabulated(parameters,spline_points, vec![0.0; max_radial]).unwrap();

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
