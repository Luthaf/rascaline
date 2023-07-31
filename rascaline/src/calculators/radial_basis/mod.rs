mod gto;
pub use self::gto::GtoRadialBasis;

mod tabulated;
pub use self::tabulated::SplinePoint;

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
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
    /// Compute the radial integral with user-defined splines.
    ///
    /// The easiest way to create a set of spline points is the
    /// `rascaline.generate_splines` Python function.
    ///
    /// For LODE calculations also the contribution of the central atom have to be
    /// provided. The `center_contribution` is defined as `c_n =
    /// \sqrt{4π} \int dr r^2 R_n(r) g(r)` where `g(r)` is a radially symmetric density
    /// function, `R_n(r)` the radial basis function and `n` the current radial channel.
    /// Note that the integration range was deliberately left ambiguous since it depends
    /// on the radial basis, i.e. for the GTO basis, `r \in R^+` is used, while `r \in
    /// [0, cutoff]` for the monomial basis.
    TabulatedRadialIntegral {
        points: Vec<SplinePoint>,
        center_contribution: Option<Vec<f64>>,
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
