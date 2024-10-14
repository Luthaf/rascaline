use std::sync::Arc;

use ndarray::Array1;

use crate::math::{HermitCubicSpline, HermitSplinePoint, SplineParameters};
use crate::Error;

/// A tabulated radial basis.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(try_from = "TabulatedSerde")]
#[serde(into = "TabulatedSerde")]
pub struct Tabulated {
    pub(crate) spline: Arc<HermitCubicSpline<ndarray::Ix1>>,
}

impl Tabulated {
    /// Get the size of the tabulated functions (i.e. how many functions are
    /// simultaneously tabulated).
    pub fn size(&self) -> usize {
        return self.spline.shape()[0]
    }
}

/// A tabulated radial basis for LODE
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(try_from = "LodeTabulatedSerde")]
#[serde(into = "LodeTabulatedSerde")]
pub struct LodeTabulated {
    pub(crate) spline: Arc<HermitCubicSpline<ndarray::Ix1>>,
    pub(crate) center_contribution: Option<Array1<f64>>,
}

impl LodeTabulated {
    /// Get the size of the tabulated functions (i.e. how many functions are
    /// simultaneously tabulated).
    pub fn size(&self) -> usize {
        return self.spline.shape()[0]
    }
}

/// Serde-compatible struct, used to serialize/deserialize splines
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TabulatedSerde {
    /// Points defining the spline
    pub points: Vec<SplinePoint>,
}

/// Serde-compatible struct, used to serialize/deserialize splines
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LodeTabulatedSerde {
    /// Points defining the spline
    pub points: Vec<SplinePoint>,
    /// The `center_contribution` is defined as `c_n = \sqrt{4π} \int dr r^2
    /// R_n(r) g(r)` where `g(r)` is a radially symmetric density function,
    /// `R_n(r)` the radial basis function and `n` the current radial channel.
    ///
    /// It is required for using tabulated basis with LODE, since we can not
    /// compute it using only the spline. This should be defined for the
    /// `lambda=0` angular channel.
    pub center_contribution: Option<Vec<f64>>,
}

/// A single point entering a spline used for the tabulated radial integrals.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SplinePoint {
    /// Position of the point
    pub position: f64,
    /// Array of values for the tabulated radial integral
    pub values: Vec<f64>,
    /// Array of derivatives for the tabulated radial integral
    pub derivatives: Vec<f64>,
}

impl TryFrom<TabulatedSerde> for Tabulated {
    type Error = Error;

    fn try_from(tabulated: TabulatedSerde) -> Result<Self, Self::Error> {
        let spline = spline_from_tabulated(tabulated.points)?;
        return Ok(Tabulated { spline });
    }
}

impl From<Tabulated> for TabulatedSerde {
    fn from(tabulated: Tabulated) -> TabulatedSerde {
        let spline = &tabulated.spline;

        let mut points = Vec::new();
        for point in &spline.points {
            points.push(SplinePoint {
                position: point.position,
                values: point.values.to_vec(),
                derivatives: point.derivatives.to_vec(),
            });
        }

        return TabulatedSerde { points };
    }
}


impl TryFrom<LodeTabulatedSerde> for LodeTabulated {
    type Error = Error;

    fn try_from(tabulated: LodeTabulatedSerde) -> Result<Self, Self::Error> {
        let spline = spline_from_tabulated(tabulated.points)?;

        let mut center_contribution = None;
        if let Some(vector) = tabulated.center_contribution {
            if vector.len() != spline.shape()[0] {
                return Err(Error::InvalidParameter(format!(
                    "expected the 'center_contribution' in 'Tabulated' \
                    radial basis to have the same number of basis function as \
                    the spline, got {} and {}",
                    vector.len(), spline.shape()[0]
                )));
            }
            center_contribution = Some(Array1::from(vector));
        }

        return Ok(LodeTabulated { spline, center_contribution });
    }
}

impl From<LodeTabulated> for LodeTabulatedSerde {
    fn from(tabulated: LodeTabulated) -> LodeTabulatedSerde {
        let spline = &tabulated.spline;

        let mut points = Vec::new();
        for point in &spline.points {
            points.push(SplinePoint {
                position: point.position,
                values: point.values.to_vec(),
                derivatives: point.derivatives.to_vec(),
            });
        }

        let center_contribution = tabulated.center_contribution.as_ref().map(Array1::to_vec);
        return LodeTabulatedSerde { points, center_contribution };
    }
}

fn spline_from_tabulated(points: Vec<SplinePoint>) -> Result<Arc<HermitCubicSpline<ndarray::Ix1>>, Error> {
    let points = check_spline_points(points)?;

    let spline_parameters = SplineParameters {
        start: points[0].position,
        stop: points[points.len() - 1].position,
        shape: vec![points[0].values.len()],
    };

    let mut new_spline_points = Vec::new();
    for point in points {
        new_spline_points.push(
            HermitSplinePoint{
                position: point.position,
                values: Array1::from(point.values),
                derivatives: Array1::from(point.derivatives),
            }
        );
    }
    let spline = Arc::new(HermitCubicSpline::new(spline_parameters, new_spline_points));

    return Ok(spline);
}

fn check_spline_points(mut points: Vec<SplinePoint>) -> Result<Vec<SplinePoint>, Error> {
    if points.len() < 2 {
        return Err(Error::InvalidParameter(
            "we need at least two points to define a 'Tabulated' radial basis".into()
        ));
    }
    let size = points[0].values.len();

    for point in &points {
        if !point.position.is_finite() {
            return Err(Error::InvalidParameter(format!(
                "expected all points 'position' in 'Tabulated' \
                radial basis to be finite numbers, got {}",
                point.position
            )));
        }

        if point.values.len() != size {
            return Err(Error::InvalidParameter(format!(
                "expected all points 'values' in 'Tabulated' \
                radial basis to have the same size, got {} and {}",
                point.values.len(), size
            )));
        }

        if point.derivatives.len() != size {
            return Err(Error::InvalidParameter(format!(
                "expected all points 'derivatives' in 'Tabulated' \
                radial basis to have the same size, got {} and {}",
                point.derivatives.len(), size
            )));
        }
    }

    points.sort_unstable_by(|a, b| a.position.total_cmp(&b.position));
    return Ok(points);
}
