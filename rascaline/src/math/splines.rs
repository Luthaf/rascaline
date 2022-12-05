use ndarray::{Array, ArrayViewMut, azip};
use log::info;

use crate::Error;


/// Maximal number of points in the splines
const MAX_SPLINE_SIZE: usize = 10_000;

/// [Hermit cubit spline][splines-wiki] implementation.
///
/// This kind of spline uses information of the value of a function on control
/// points as well as the gradient of the function at these points. This
/// implementation takes a single scalar as input, and output array values, i.e.
/// we can spline functions of the form `R -> R^n`.
///
/// [splines-wiki]: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
#[derive(Debug, Clone)]
pub struct HermitCubicSpline<D: ndarray::Dimension> {
    parameters: SplineParameters,
    points: Vec<HermitSplinePoint<D>>,
}


/// Parameters controlling a `HermitCubicSpline`
#[derive(Debug, Clone)]
pub struct SplineParameters {
    /// Beginning of the interpolation space
    pub start: f64,
    /// End of the interpolation space
    pub stop: f64,
    /// Shape of the array output
    pub shape: Vec<usize>,
}

/// A single control point/knot in the Hermit cubic spline
#[derive(Debug, Clone)]
pub struct HermitSplinePoint<D: ndarray::Dimension> {
    /// Position of the point
    pub(crate) position: f64,
    /// Value of the function to interpolate at the position
    pub(crate) value: Array<f64, D>,
    /// Derivative of the function to interpolate at the position
    pub(crate) derivative: Array<f64, D>,
}

impl<D: ndarray::Dimension> HermitCubicSpline<D> {
    #[allow(clippy::float_cmp)]
    pub(crate) fn new(
        parameters: SplineParameters,
        mut points: Vec<HermitSplinePoint<D>>,
    ) -> HermitCubicSpline<D>  {
        assert!(points.len() >= 2, "we need at least two points to create a spline");
        assert!(parameters.start < parameters.stop);

        points.sort_unstable_by(|a, b| {
            a.position.partial_cmp(&b.position).expect("got NaN while sorting by point position")
        });

        assert_eq!(points.first().unwrap().position, parameters.start);
        assert!(points.last().unwrap().position >= parameters.stop);

        Self {
            parameters: parameters,
            points: points,
        }
    }

    /// Create a new `HermitCubicSpline` from the given function, trying to
    /// reach the given accuracy on average.
    ///
    /// When called, the `function` should return a tuple of `(value, gradient)`
    /// at the input position, where `value` and `gradient` are arrays with the
    /// same shape as `parameter.shape`.
    ///
    /// Points are added to the spline until the requested accuracy is reached.
    /// We consider that the accuracy is reached when either the mean absolute
    /// error or the mean relative error gets below the `accuracy` threshold.
    pub fn with_accuracy<F>(
        accuracy: f64,
        parameters: SplineParameters,
        function: F,
    ) -> Result<HermitCubicSpline<D>, Error> where
            F: Fn(f64) -> (Array<f64, D>, Array<f64, D>),
    {
        if accuracy < 0.0 {
            return Err(Error::InvalidParameter(format!(
                "got invalid accuracy in spline ({}), it must be positive", accuracy
            )));
        }

        let interpolated = Array::from_elem(parameters.shape.clone(), 0.0);
        let mut interpolated = match interpolated.into_dimensionality::<D>() {
            Ok(array) => array,
            Err(e) => {
                return Err(Error::InvalidParameter(format!(
                    "wrong shape parameter: {}", e
                )));
            }
        };

        let initial_grid_size = 11;
        let grid_step = (parameters.stop - parameters.start) / (initial_grid_size - 1) as f64;

        let mut points = Vec::new();
        for k in 0..initial_grid_size {
            let position = parameters.start + k as f64 * grid_step;
            let (value, derivative) = function(position);

            if value.shape() != parameters.shape || derivative.shape() != parameters.shape  {
                return Err(Error::InvalidParameter(format!(
                    "function ({:?}) or gradient of the function ({:?}) returned a different shape than expected ({:?})",
                    value.shape(), derivative.shape(), parameters.shape
                )));
            }

            points.push(HermitSplinePoint { position, value, derivative });
        }

        let mut spline = HermitCubicSpline::new(parameters, points);

        // add more points as required to reach the requested accuracy
        loop {
            let mut max_absolute_error = 0.0;
            let mut mean_absolute_error = 0.0;
            let mut mean_relative_error = 0.0;
            let mut error_count = 0;

            let positions = spline.positions();

            // evaluate the error at points in between grid points, since these
            // should have the highest error in average.
            let mut new_points = Vec::new();
            for k in 0..(spline.len() - 1) {
                let position = (positions[k] + positions[k + 1]) / 2.0;

                let (value, derivative) = function(position);

                interpolated.fill(0.0);
                spline.compute(position, interpolated.view_mut(), None);

                // get the error across all values in the arrays
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
                    "spline reached requested accuracy ({:.3e}) with {} reference points (max absolute error is {:.3e})",
                    accuracy, spline.len(), max_absolute_error,
                );
                break;
            }

            if spline.len() + new_points.len() > MAX_SPLINE_SIZE {
                return Err(Error::Internal(format!(
                    "failed to reach requested accuracy ({:e}) in spline interpolation, \
                    mean absolute error is {:e} and mean relative error is {:e}",
                    accuracy, mean_absolute_error, mean_relative_error
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
    /// `self.start` and `self.stop`.
    fn add_point(&mut self, point: HermitSplinePoint<D>) {
        debug_assert!(point.position > self.parameters.start);
        debug_assert!(point.position < self.parameters.stop);
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

    /// Compute the spline at point `x`, storing the results in `values` and
    /// optionally `gradients`.
    pub fn compute(&self, x: f64, values: ArrayViewMut<f64, D>, gradients: Option<ArrayViewMut<f64, D>>) {
        debug_assert!(x.is_finite());
        debug_assert!(x >= self.parameters.start && x < self.parameters.stop);
        debug_assert_eq!(values.shape(), self.parameters.shape);
        if let Some(ref gradients) = gradients {
            debug_assert_eq!(gradients.shape(), self.parameters.shape);
        }


        // notation in this function follows
        // https://en.wikipedia.org/wiki/Cubic_Hermite_spline

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
    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};

    #[test]
    fn evaluate_simple_spline() {
        let accuracy = 1e-9;
        let parameters = SplineParameters {
            start: -3.0,
            stop: 6.0,
            shape: vec![1],
        };
        let spline = HermitCubicSpline::with_accuracy(
            accuracy,
            parameters,
            |x| (ndarray::arr1(&[f64::sin(x)]), ndarray::arr1(&[f64::cos(x)])),
        ).unwrap();

        let mut values = ndarray::Array1::from_elem((1,), 0.0);
        let mut gradients = ndarray::Array1::from_elem((1,), 0.0);
        for &x in &[-2.2, -1.00242144, 0.0, 0.000000001, 2.3, 3.2, 4.7, 5.3, 5.99999999] {
            spline.compute(x, values.view_mut(), Some(gradients.view_mut()));
            assert_relative_eq!(values[0], f64::sin(x), max_relative=1e-5, epsilon=1e-12);
            assert_relative_eq!(gradients[0], f64::cos(x), max_relative=1e-5, epsilon=1e-12);
        }

        // check that the values match exactly at the control points. The only
        // exception is the last control point were we can not compute the
        // spline.
        for &x in spline.positions().iter().filter(|&&x| x < spline.parameters.stop) {
            spline.compute(x, values.view_mut(), Some(gradients.view_mut()));
            assert_ulps_eq!(values[0], f64::sin(x));
            assert_ulps_eq!(gradients[0], f64::cos(x));
        }
    }

    #[test]
    #[should_panic = "got invalid accuracy in spline (-1), it must be positive"]
    fn invalid_accuracy() {
        let parameters = SplineParameters {
            start: -3.0,
            stop: 1.2,
            shape: vec![1],
        };

        HermitCubicSpline::with_accuracy(
            -1.0,
            parameters,
            |x| (ndarray::arr1(&[f64::sin(x)]), ndarray::arr1(&[f64::cos(x)])),
        ).unwrap();
    }
}
