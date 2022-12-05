mod gto;
pub use self::gto::GtoRadialBasis;
use ndarray::{Array2};

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
/// Radial basis that can be used in the SOAP or LODE spherical expansion
pub enum RadialBasis {
    /// Use a radial basis similar to Gaussian-Type Orbitals.
    ///
    /// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
    /// = cutoff * \sqrt{n} / n_max`
    Gto {
        /// compute the radial integral using splines. This is much faster than
        /// the base GTO implementation.
        #[serde(default = "serde_default_splined_radial_integral")]
        splined_radial_integral: bool,
        /// Accuracy for the spline. The number of control points in the spline
        /// is automatically determined to ensure the average absolute error is
        /// close to the requested accuracy.
        #[serde(default = "serde_default_spline_accuracy")]
        spline_accuracy: f64,
    },
    TabulatedRadialIntegral {
        /// Provide user-defined splines. These consist of the positions of the spline 
        /// points, values of the radial integrals at the spline point for each l 
        /// and each n, and derivatives of the radial integrals at the spline 
        /// points for each l and each n.
        spline_points: Vec<SplinePoint>,
    }
}

#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SplinePoint {
    pub(crate) position: f64,
    pub(crate) values: Array2<f64>,
    pub(crate) derivatives: Array2<f64>,
}

impl schemars::JsonSchema for RadialBasis {
    fn schema_name() -> String {
       todo!()
    }
 
    fn json_schema(_gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
       todo!()
    }
 }

fn serde_default_splined_radial_integral() -> bool { true }
fn serde_default_spline_accuracy() -> f64 { 1e-8 }

impl RadialBasis {
    /// Use GTO as the radial basis, and do not spline the radial integral
    pub fn gto() -> RadialBasis {
        return RadialBasis::Gto {
            splined_radial_integral: false, spline_accuracy: 0.0
        };
    }

    /// Use GTO as the radial basis, and spline the radial integral
    pub fn splined_gto(accuracy: f64) -> RadialBasis {
        return RadialBasis::Gto {
            splined_radial_integral: true, spline_accuracy: accuracy
        };
    }
}
