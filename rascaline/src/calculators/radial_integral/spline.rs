use ndarray::{Array2, ArrayViewMut2, azip};
use log::info;

use super::RadialIntegral;
use crate::Error;

/// Maximal number of points in the splines
const MAX_SPLINE_SIZE: usize = 10_000;

/// `SplinedRadialIntegral` allows to evaluate another radial integral
/// implementation using [cubic Hermit spline][splines-wiki].
///
/// This can be much faster than using the actual radial integral
/// implementation.
///
/// [splines-wiki]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
pub struct SplinedRadialIntegral {
    parameters: SplinedRIParameters,
    points: Vec<HermitSplinePoint>,
}

/// A single control point/knot in the Hermit cubic spline
#[derive(Debug, Clone)]
struct HermitSplinePoint {
    /// Position of the point
    position: f64,
    /// Value of the function to interpolate at the position
    value: Array2<f64>,
    /// Derivative of the function to interpolate at the position
    derivative: Array2<f64>,
}

/// Parameters for computing the radial integral using Hermit cubic splines
#[derive(Debug, Clone, Copy)]
pub struct SplinedRIParameters {
    /// Number of radial components
    pub max_radial: usize,
    /// Number of angular components
    pub max_angular: usize,
    /// cutoff radius, this is also the maximal value that can be interpolated
    pub cutoff: f64,
}

impl SplinedRadialIntegral {
    #[allow(clippy::float_cmp)]
    fn new(
        parameters: SplinedRIParameters,
        mut points: Vec<HermitSplinePoint>,
    ) -> SplinedRadialIntegral  {
        assert!(points.len() >= 2, "we need at least two points to create a spline");

        points.sort_unstable_by(|a, b| {
            a.position.partial_cmp(&b.position).expect("got NaN while sorting by point position")
        });

        assert_eq!(points.first().unwrap().position, 0.0);
        assert_eq!(points.last().unwrap().position, parameters.cutoff);

        SplinedRadialIntegral {
            parameters: parameters,
            points: points,
        }
    }

    /// Create a new `SplinedRadialIntegral` taking values from the given
    /// `radial_integral`. Points are added to the spline until the requested
    /// accuracy is reached. We consider that the accuracy is reached when
    /// either the mean absolute error or the mean relative error gets below the
    /// `accuracy` threshold.
    #[time_graph::instrument(name = "SplinedRadialIntegral::with_accuracy")]
    pub fn with_accuracy(
        parameters: SplinedRIParameters,
        accuracy: f64,
        radial_integral: impl RadialIntegral
    ) -> Result<SplinedRadialIntegral, Error> {
        if accuracy < 0.0 {
            return Err(Error::InvalidParameter(format!(
                "got invalid accuracy in spline ({}), it must be positive", accuracy
            )));
        }

        let initial_grid_size = 11;
        let grid_step = parameters.cutoff / (initial_grid_size - 1) as f64;

        let mut points = Vec::new();
        let shape = (parameters.max_angular + 1, parameters.max_radial);
        for k in 0..initial_grid_size {
            let position = k as f64 * grid_step;
            let mut value = Array2::from_elem(shape, 0.0);
            let mut derivative = Array2::from_elem(shape, 0.0);
            radial_integral.compute(position, value.view_mut(), Some(derivative.view_mut()));

            points.push(HermitSplinePoint { position, value, derivative });
        }

        let mut spline = SplinedRadialIntegral::new(parameters, points);

        // add more points as required to reach the requested accuracy
        loop {
            let mut new_points = Vec::new();

            let positions = spline.positions();

            // evaluate the error at points in between grid points, since these
            // should have the highest error in average.
            let mut max_absolute_error = 0.0;
            let mut mean_absolute_error = 0.0;
            let mut mean_relative_error = 0.0;
            let mut error_count = 0;
            for k in 0..(spline.len() - 1) {
                let position = (positions[k] + positions[k + 1]) / 2.0;

                let mut value = Array2::from_elem(shape, 0.0);
                let mut derivative = Array2::from_elem(shape, 0.0);
                radial_integral.compute(position, value.view_mut(), Some(derivative.view_mut()));

                let mut interpolated = Array2::from_elem(shape, 0.0);
                spline.compute(position, interpolated.view_mut(), None);

                // get the error across all n/l values in the arrays
                azip!((interpolated in &interpolated, value in &value) {
                    let absolute_error = f64::abs(interpolated - value);
                    if absolute_error > max_absolute_error {
                        max_absolute_error = absolute_error;
                    }

                    mean_absolute_error += absolute_error;
                    mean_relative_error += f64::abs((interpolated - value) / value);
                    error_count += 1;
                });

                new_points.push(HermitSplinePoint { position, value, derivative });
            }
            mean_absolute_error /= error_count as f64;
            mean_relative_error /= error_count as f64;

            if mean_absolute_error < accuracy || mean_relative_error < accuracy {
                info!(
                    "splined radial integral reached requested accuracy ({:.3e}) on average with {} reference points (max absolute error is {:.3e})",
                    accuracy, spline.len(), max_absolute_error,
                );
                break;
            }

            if spline.len() + new_points.len() > MAX_SPLINE_SIZE {
                return Err(Error::Internal(format!(
                    "failed to reach requested accuracy ({:e}) in spline interpolation for radial integral, \
                    the best we got was {:e}",
                    accuracy, max_absolute_error
                )));
            }

            // add more points and continue
            for point in new_points {
                spline.add_point(point);
            }
        }

        return Ok(spline);
    }

    /// Add a new control points to this spline. The new point must be between
    /// 0 and the cutoff.
    fn add_point(&mut self, point: HermitSplinePoint) {
        debug_assert!(point.position > 0.0 && point.position < self.parameters.cutoff );
        match self.points.binary_search_by(
            |v| v.position.partial_cmp(&point.position).expect("got NaN")
        ) {
            Ok(_) => panic!("trying to add the same point twice to the spline"),
            Err(k) => self.points.insert(k, point)
        }
    }

    /// Get the number of control points in this spline
    fn len(&self) -> usize {
        self.points.len()
    }

    /// Get the position of the control points for this spline
    fn positions(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.position).collect()
    }
}

impl RadialIntegral for SplinedRadialIntegral {
    #[time_graph::instrument(name = "SplinedRadialIntegral::compute")]
    fn compute(&self, x: f64, values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>) {
        // notation in this function follows
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        debug_assert!(x < self.parameters.cutoff && x >= 0.0 && x.is_finite());

        let k = match self.points.binary_search_by(
            |v| v.position.partial_cmp(&x).expect("got NaN")
        ) {
            Ok(k) => k,
            Err(k) => k - 1,
        };

        let point_k = &self.points[k];
        let point_k_1 = &self.points[k + 1];

        let x_k = point_k.position;
        let x_k_1 = point_k_1.position;
        debug_assert!(x_k <= x && x < x_k_1);

        let delta = x_k_1 - x_k;
        let t = (x - x_k) / delta;
        let t_2 = t * t;
        let t_3 = t_2 * t;

        // Hermit base polynomials
        let h00 = 2.0 * t_3 - 3.0 * t_2 + 1.0;
        let h10 = t_3 - 2.0 * t_2 + t;
        let h01 = -2.0 * t_3 + 3.0 * t_2;
        let h11 = t_3 - t_2;

        let p_k = &point_k.value;
        let p_k_1 = &point_k_1.value;

        let m_k = &point_k.derivative;
        let m_k_1 = &point_k_1.derivative;

        azip!((v in values, p_k in p_k, p_k_1 in p_k_1, m_k in m_k, m_k_1 in m_k_1) {
            *v = h00 * p_k + h10 * delta * m_k + h01 * p_k_1 + h11 * delta * m_k_1;
        });

        if let Some(gradients) = gradients {
            let d_h00_dt = 6.0 * (t_2 - t);
            let d_h10_dt = 3.0 * t_2 - 4.0 * t + 1.0;
            let d_h01_dt = -d_h00_dt;
            let d_h11_dt = 3.0 * t_2 - 2.0 * t;

            let dx_dt = 1.0 / delta;

            azip!((g in gradients, p_k in p_k, p_k_1 in p_k_1, m_k in m_k, m_k_1 in m_k_1) {
                *g = d_h00_dt * p_k * dx_dt + d_h10_dt * m_k + d_h01_dt * p_k_1 * dx_dt + d_h11_dt * m_k_1;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, assert_ulps_eq};

    use super::*;
    use super::super::super::soap::{SoapGtoRadialIntegral, GtoParameters};

    struct FalseRadialIntegral;
    impl RadialIntegral for FalseRadialIntegral {
        fn compute(&self, x: f64, mut values: ArrayViewMut2<f64>, gradients: Option<ArrayViewMut2<f64>>) {
            values[[0, 0]] = f64::sin(x);
            if let Some(mut gradients) = gradients {
                gradients[[0, 0]] = f64::cos(x);
            }
        }
    }

    #[test]
    fn evaluate_simple_spline() {
        let accuracy = 1e-9;
        let cutoff = 6.0;
        let parameters = SplinedRIParameters {
            max_radial: 1,
            max_angular: 0,
            cutoff: cutoff,
        };
        let shape = (1, 1);
        let spline = SplinedRadialIntegral::with_accuracy(
            parameters, accuracy, FalseRadialIntegral
        ).unwrap();

        for &x in &[0.0, 0.000000001, 2.3, 3.2, 4.7, 5.3, 5.99999999] {
            let mut values = Array2::from_elem(shape, 0.0);
            let mut gradients = Array2::from_elem(shape, 0.0);

            spline.compute(x, values.view_mut(), Some(gradients.view_mut()));
            assert_relative_eq!(values[[0, 0]], f64::sin(x), max_relative=1e-5);
            assert_relative_eq!(gradients[[0, 0]], f64::cos(x), max_relative=1e-5);
        }

        // check that the values match exactly at the control points. The only
        // exception is the last control point (i.e. the cutoff) were we can not
        // compute the spline.
        for &x in spline.positions().iter().filter(|&&x| x < cutoff) {
            let mut values = Array2::from_elem(shape, 0.0);
            let mut gradients = Array2::from_elem(shape, 0.0);

            spline.compute(x, values.view_mut(), Some(gradients.view_mut()));
            assert_ulps_eq!(values[[0, 0]], f64::sin(x));
            assert_ulps_eq!(gradients[[0, 0]], f64::cos(x));
        }
    }

    #[test]
    #[should_panic = "got invalid accuracy in spline (-1), it must be positive"]
    fn invalid_accuracy() {
        let parameters = SplinedRIParameters {
            max_radial: 1,
            max_angular: 0,
            cutoff: 6.0,
        };
        SplinedRadialIntegral::with_accuracy(parameters, -1.0, FalseRadialIntegral).unwrap();
    }

    #[test]
    fn high_accuracy() {
        // Check that even with high accuracy and large domain MAX_SPLINE_SIZE
        // is enough
        let parameters = SplinedRIParameters {
            max_radial: 15,
            max_angular: 10,
            cutoff: 12.0,
        };

        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        // this test only check that this code runs without crashing
        SplinedRadialIntegral::with_accuracy(parameters, 1e-10, gto).unwrap();
    }

    #[test]
    fn finite_difference() {
        let max_radial = 8;
        let max_angular = 8;
        let parameters = SplinedRIParameters {
            max_radial: max_radial,
            max_angular: max_angular,
            cutoff: 5.0,
        };

        let gto = SoapGtoRadialIntegral::new(GtoParameters {
            max_radial: parameters.max_radial,
            max_angular: parameters.max_angular,
            cutoff: parameters.cutoff,
            atomic_gaussian_width: 0.5,
        }).unwrap();

        // even with very bad accuracy, we want the gradients of the spline to
        // match the values produces by the spline, and not necessarily the
        // actual GTO gradients.
        let spline = SplinedRadialIntegral::with_accuracy(parameters, 1e-2, gto).unwrap();

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
}
