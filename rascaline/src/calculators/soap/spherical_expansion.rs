use ndarray::Array2;

use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, SamplesIndexes, TwoBodiesSpeciesSamples};
use crate::{Descriptor, Error, System, Vector3D};

use super::super::CalculatorBase;
use super::RadialIntegral;
use super::{GtoRadialIntegral, GtoParameters};
use super::{SplinedRadialIntegral, SplinedRIParameters};

use super::{SphericalHarmonics, SphericalHarmonicsArray};

/// Specialized function to compute (-1)^l. Using this instead of
/// `f64::powi(-1.0, l as i32)` shaves 10% of the computational time
fn m_1_pow(l: usize) -> f64 {
    if l % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// Radial basis that can be used in the spherical expansion
pub enum RadialBasis {
    /// Use a radial basis similar to Gaussian-Type Orbitals.
    ///
    /// The basis is defined as `R_n(r) ∝ r^n e^{- r^2 / (2 σ_n^2)}`, where `σ_n
    /// = cutoff * \sqrt{n} / n_max`
    Gto {},
    /// Splined version of the `Gto` radial basis.
    ///
    /// This computes the same integral as the GTO radial basis but using Cubic
    /// Hermit splines with control points sampled from the GTO implementation.
    /// Using splines is usually much faster (up to 30% of the runtime in the
    /// spherical expansion) than using the base GTO implementation.
    ///
    /// The number of control points in the spline is automatically determined
    /// to ensure the maximal absolute error is close to the requested accuracy.
    SplinedGto {
        accuracy: f64,
    },
}

impl RadialBasis {
    fn construct(&self, parameters: &SphericalExpansionParameters) -> Result<Box<dyn RadialIntegral>, Error> {
        match self {
            RadialBasis::Gto {} => {
                let parameters = GtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                return Ok(Box::new(GtoRadialIntegral::new(parameters)?));
            }
            RadialBasis::SplinedGto { accuracy } => {
                let parameters = GtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                let gto = GtoRadialIntegral::new(parameters)?;

                let parameters = SplinedRIParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    cutoff: parameters.cutoff,
                };
                return Ok(Box::new(SplinedRadialIntegral::with_accuracy(parameters, *accuracy, gto)?));
            }
        };
    }
}

/// Possible values for the smoothing cutoff function
#[derive(Debug, Clone, Copy)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub enum CutoffFunction {
    /// Step function, 1 if `r < cutoff` and 0 if `r >= cutoff`
    Step{},
    /// Shifted cosine switching function
    /// `f(r) = 1/2 * (1 + cos(π (r - cutoff + width) / width ))`
    ShiftedCosine {
        width: f64,
    },
}

impl CutoffFunction {
    /// Evaluate the cutoff function at the distance `r` for the given `cutoff`
    pub fn compute(&self, r: f64, cutoff: f64) -> f64 {
        match self {
            CutoffFunction::Step{} => {
                if r >= cutoff { 0.0 } else { 1.0 }
            },
            CutoffFunction::ShiftedCosine { width } => {
                if r <= (cutoff - width) {
                    1.0
                } else if r >= cutoff {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - cutoff + width) / width;
                    0.5 * (1. + f64::cos(s))
                }
            }
        }
    }

    /// Evaluate the derivative of the cutoff function at the distance `r` for the
    /// given `cutoff`
    pub fn derivative(&self, r: f64, cutoff: f64) -> f64 {
        match self {
            CutoffFunction::Step{} => 0.0,
            CutoffFunction::ShiftedCosine { width } => {
                if r <= (cutoff - width) || r >= cutoff {
                    0.0
                } else {
                    let s = std::f64::consts::PI * (r - cutoff + width) / width;
                    return -0.5 * std::f64::consts::PI * f64::sin(s) / width;
                }
            }
        }
    }
}

/// Parameters for spherical expansion calculator.
///
/// The spherical expansion is at the core of representations in the SOAP
/// (Smooth Overlap of Atomic Positions) family. See [this review
/// article](https://doi.org/10.1063/1.5090481) for more information on the SOAP
/// representation, and [this paper](https://doi.org/10.1063/5.0044689) for
/// information on how it is implemented in rascaline.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[allow(clippy::module_name_repetitions)]
pub struct SphericalExpansionParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,
    /// Should we also compute gradients of the feature?
    pub gradients: bool,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
}

struct RadialIntegralImpl {
    /// Implementation of the radial integral
    code: Box<dyn RadialIntegral>,
    /// Cache for the radial integral values
    values: Array2<f64>,
    /// Cache for the radial integral gradient
    gradients: Option<Array2<f64>>,
}

impl RadialIntegralImpl {
    fn new(parameters: &SphericalExpansionParameters) -> Result<Self, Error> {
        let code = parameters.radial_basis.construct(&parameters)?;
        let shape = (parameters.max_radial, parameters.max_angular + 1);
        let values = Array2::from_elem(shape, 0.0);
        let gradients = if parameters.gradients {
            Some(Array2::from_elem(shape, 0.0))
        } else {
            None
        };

        return Ok(RadialIntegralImpl { code, values, gradients });
    }

    fn compute(&mut self, distance: f64) {
        self.code.compute(
            distance,
            self.values.view_mut(),
            self.gradients.as_mut().map(|o| o.view_mut()),
        );
    }

    fn compute_no_gradients(&mut self, distance: f64) {
        self.code.compute(distance, self.values.view_mut(), None);
    }
}

struct SphericalHarmonicsImpl {
    /// Implementation of the spherical harmonics
    code: SphericalHarmonics,
    /// Cache for the spherical harmonics values
    values: SphericalHarmonicsArray,
    /// Cache for the spherical harmonics gradients (one value each for x/y/z)
    gradients: Option<[SphericalHarmonicsArray; 3]>,
}

impl SphericalHarmonicsImpl {
    fn new(parameters: &SphericalExpansionParameters) -> SphericalHarmonicsImpl {
        let code = SphericalHarmonics::new(parameters.max_angular);
        let values = SphericalHarmonicsArray::new(parameters.max_angular);
        let gradients = if parameters.gradients {
            Some([
                SphericalHarmonicsArray::new(parameters.max_angular),
                SphericalHarmonicsArray::new(parameters.max_angular),
                SphericalHarmonicsArray::new(parameters.max_angular)
            ])
        } else {
            None
        };

        return SphericalHarmonicsImpl { code, values, gradients };
    }

    fn compute(&mut self, direction: Vector3D) {
        self.code.compute(
            direction,
            &mut self.values,
            self.gradients.as_mut(),
        );
    }

    fn compute_no_gradients(&mut self, direction: Vector3D) {
        self.code.compute(direction, &mut self.values, self.gradients.as_mut());
    }
}

/// The actual calculator used to compute SOAP spherical expansion coefficients
pub struct SphericalExpansion {
    /// Parameters governing the spherical expansion
    parameters: SphericalExpansionParameters,
    radial_integral: RadialIntegralImpl,
    spherical_harmonics: SphericalHarmonicsImpl,
}

impl std::fmt::Debug for SphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SphericalExpansion {
    /// Create a new `SphericalExpansion` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansion, Error> {
        let radial_integral = RadialIntegralImpl::new(&parameters)?;
        let spherical_harmonics = SphericalHarmonicsImpl::new(&parameters);

        return Ok(SphericalExpansion {
            parameters,
            radial_integral,
            spherical_harmonics
        });
    }

    /// Compute the self contribution to spherical expansion, i.e. the
    /// contribution of the central atom own density to the expansion around
    /// itself.
    ///
    /// The self contribution does not have contributions to the gradients
    fn do_self_contributions(&mut self, descriptor: &mut Descriptor) {
        // we could cache the self contribution since they only depend on the
        // gaussian atomic width. For now, we recompute them all the time

        for (i_env, requested_env) in descriptor.samples.iter().enumerate() {
            let species_center = requested_env[2];
            let species_neighbor = requested_env[3];

            if species_center == species_neighbor {
                self.radial_integral.compute_no_gradients(0.0);
                self.spherical_harmonics.compute_no_gradients(Vector3D::new(0.0, 0.0, 1.0));
                let f_cut = self.parameters.cutoff_function.compute(0.0, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut
                        * self.radial_integral.values[[n, l]]
                        * self.spherical_harmonics.values[[l as isize, m]];
                    descriptor.values[[i_env, feature_i]] += n_l_m_value;
                }
            }
        }
    }

    /// Accumulate the spherical expansion coefficients for the given pair.
    ///
    /// This function assumes that the radial integral and spherical harmonics
    /// have just been computed for this pair.
    fn accumulate_for_pair(&mut self, descriptor: &mut Descriptor, pair: &Pair) {
        let first_sample_i = descriptor.samples.position(&[
            IndexValue::from(pair.system),
            IndexValue::from(pair.first),
            IndexValue::from(pair.species_first),
            IndexValue::from(pair.species_second),
        ]);

        let second_sample_i;
        if pair.first == pair.second {
            // do not compute for the reversed pair if the pair is between an
            // atom and its image
            second_sample_i = None;
        } else {
            second_sample_i = descriptor.samples.position(&[
                IndexValue::from(pair.system),
                IndexValue::from(pair.second),
                IndexValue::from(pair.species_second),
                IndexValue::from(pair.species_first),
            ]);
        }

        if first_sample_i.is_none() && second_sample_i.is_none() {
            // nothing to do
            return;
        }

        self.radial_integral.compute(pair.distance);
        self.spherical_harmonics.compute(pair.direction);

        let f_cut = self.parameters.cutoff_function.compute(
            pair.distance, self.parameters.cutoff
        );

        for (feature_i, feature) in descriptor.features.iter().enumerate() {
            let n = feature[0].usize();
            let l = feature[1].usize();
            let m = feature[2].isize();

            let n_l_m_value = f_cut
                * self.radial_integral.values[[n, l]]
                * self.spherical_harmonics.values[[l as isize, m]];
            if let Some(first_sample_i) = first_sample_i {
                descriptor.values[[first_sample_i, feature_i]] += n_l_m_value;
            }

            if let Some(second_sample_i) = second_sample_i {
                // Use the fact that `se[n, l, m](-r) = (-1)^l se[n, l, m](r)`
                // where se === spherical_expansion.
                descriptor.values[[second_sample_i, feature_i]] += m_1_pow(l) * n_l_m_value;
            }
        }
    }

    /// Accumulate the spherical expansion gradients for the given pair.
    ///
    /// This function assumes that the radial integral and spherical harmonics
    /// have just been computed for this pair.
    #[allow(clippy::similar_names, clippy::identity_op, clippy::clippy::too_many_lines)]
    fn accumulate_gradient_for_pair(&self, descriptor: &mut Descriptor, pair: &Pair) {
        debug_assert!(self.parameters.gradients);

        let gradients = descriptor.gradients.as_mut().expect("missing storage for gradients");
        let gradients_samples = descriptor.gradients_samples.as_ref().expect("missing gradients samples");

        // get the positions in the gradients array where to store the
        // contributions to the gradient for this specific pair, if they
        // exist in the set of requested samples

        // store derivative w.r.t. `second` of the environment around `first`
        let first_grad_i = gradients_samples.position(&[
            IndexValue::from(pair.system),
            IndexValue::from(pair.first),
            IndexValue::from(pair.species_first),
            IndexValue::from(pair.species_second),
            IndexValue::from(pair.second),
            IndexValue::from(0)
        ]);

        // store derivative w.r.t. `first` of the environment around `first`
        let first_self_grad_i = gradients_samples.position(&[
            IndexValue::from(pair.system),
            IndexValue::from(pair.first),
            IndexValue::from(pair.species_first),
            IndexValue::from(pair.species_second),
            IndexValue::from(pair.first),
            IndexValue::from(0),
        ]);

        // store derivative w.r.t. `first` of the environment around `second`
        let second_grad_i;
        // store derivative w.r.t. `second` of the environment around `second`
        let second_self_grad_i;
        if pair.first == pair.second {
            // do not compute for the reversed pair if the pair is between an
            // atom and its image
            second_grad_i = None;
            second_self_grad_i = None;
        } else {
            second_grad_i = gradients_samples.position(&[
                IndexValue::from(pair.system),
                IndexValue::from(pair.second),
                IndexValue::from(pair.species_second),
                IndexValue::from(pair.species_first),
                IndexValue::from(pair.first),
                IndexValue::from(0)
            ]);

            second_self_grad_i = gradients_samples.position(&[
                IndexValue::from(pair.system),
                IndexValue::from(pair.second),
                IndexValue::from(pair.species_second),
                IndexValue::from(pair.species_first),
                IndexValue::from(pair.second),
                IndexValue::from(0),
            ]);
        }

        if first_grad_i.is_none() && first_self_grad_i.is_none()
           && second_grad_i.is_none() && second_self_grad_i.is_none()  {
            // nothing to do
            return;
        }

        let ri_values = &self.radial_integral.values;
        let ri_gradients = self.radial_integral.gradients.as_ref().expect("missing radial integral gradients");

        let sph_values = &self.spherical_harmonics.values;
        let sph_gradients = self.spherical_harmonics.gradients.as_ref().expect("missing spherical harmonics gradients");

        let f_cut = self.parameters.cutoff_function.compute(
            pair.distance, self.parameters.cutoff
        );
        let f_cut_grad = self.parameters.cutoff_function.derivative(
            pair.distance, self.parameters.cutoff
        );

        let dr_dx = pair.direction[0];
        let dr_dy = pair.direction[1];
        let dr_dz = pair.direction[2];

        for (feature_i, feature) in descriptor.features.iter().enumerate() {
            let n = feature[0].usize();
            let l = feature[1].usize();
            let m = feature[2].isize();

            let sph_value = sph_values[[l as isize, m]];
            let sph_grad_x = sph_gradients[0][[l as isize, m]];
            let sph_grad_y = sph_gradients[1][[l as isize, m]];
            let sph_grad_z = sph_gradients[2][[l as isize, m]];

            let ri_value = ri_values[[n, l]];
            let ri_grad = ri_gradients[[n, l]];

            let grad_x = f_cut_grad * dr_dx * ri_value * sph_value
                        + f_cut * ri_grad * dr_dx * sph_value
                        + f_cut * ri_value * sph_grad_x / pair.distance;

            let grad_y = f_cut_grad * dr_dy * ri_value * sph_value
                        + f_cut * ri_grad * dr_dy * sph_value
                        + f_cut * ri_value * sph_grad_y / pair.distance;

            let grad_z = f_cut_grad * dr_dz * ri_value * sph_value
                        + f_cut * ri_grad * dr_dz * sph_value
                        + f_cut * ri_value * sph_grad_z / pair.distance;

            // we assume that the three spatial derivative are stored
            // one after the other
            if let Some(grad_i) = first_grad_i {
                gradients[[grad_i + 0, feature_i]] += grad_x;
                gradients[[grad_i + 1, feature_i]] += grad_y;
                gradients[[grad_i + 2, feature_i]] += grad_z;
            }

            if let Some(grad_i) = first_self_grad_i {
                gradients[[grad_i + 0, feature_i]] -= grad_x;
                gradients[[grad_i + 1, feature_i]] -= grad_y;
                gradients[[grad_i + 2, feature_i]] -= grad_z;
            }

            // when storing data for the environment around `second`, use
            // the fact that
            // `grad_j se_i[n, l, m](r) = - (-1)^l grad_i se_j[n, l, m](r)`
            // where se === spherical_expansion.
            let parity = m_1_pow(l);
            let second_grad_x = - parity * grad_x;
            let second_grad_y = - parity * grad_y;
            let second_grad_z = - parity * grad_z;

            if let Some(grad_i) = second_grad_i {
                gradients[[grad_i + 0, feature_i]] += second_grad_x;
                gradients[[grad_i + 1, feature_i]] += second_grad_y;
                gradients[[grad_i + 2, feature_i]] += second_grad_z;
            }

            if let Some(grad_i) = second_self_grad_i {
                gradients[[grad_i + 0, feature_i]] -= second_grad_x;
                gradients[[grad_i + 1, feature_i]] -= second_grad_y;
                gradients[[grad_i + 2, feature_i]] -= second_grad_z;
            }
        }
    }
}

/// Pair data for spherical expansion, with a bit more data than the system
/// pairs
#[derive(Debug, Clone)]
struct Pair {
    /// index of the system this pair comes from
    system: usize,
    /// index of the first atom of the pair inside the system
    first: usize,
    /// index of the second atom of the pair inside the system
    second: usize,
    /// species of the first atom of the pair
    species_first: usize,
    /// species of the second atom of the pair
    species_second: usize,
    /// distance between the first and second atom in the pair
    distance: f64,
    /// direction vector (normalized) from the first to the second atom in the
    /// pair
    direction: Vector3D,
}

impl CalculatorBase for SphericalExpansion {
    fn name(&self) -> String {
        "spherical expansion".into()
    }

    fn get_parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn features_names(&self) -> Vec<&str> {
        vec!["n", "l", "m"]
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(self.features_names());
        for n in 0..(self.parameters.max_radial as isize) {
            for l in 0..((self.parameters.max_angular + 1) as isize) {
                for m in -l..=l {
                    features.add(&[
                        IndexValue::from(n), IndexValue::from(l), IndexValue::from(m)
                    ]);
                }
            }
        }
        return features.finish();
    }

    fn samples(&self) -> Box<dyn SamplesIndexes> {
        Box::new(TwoBodiesSpeciesSamples::with_self_contribution(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        assert_eq!(indexes.names(), self.features_names());
        for value in indexes {
            let n = value[0].usize();
            let l = value[1].isize();
            let m = value[2].isize();

            if n >= self.parameters.max_radial {
                return Err(Error::InvalidParameter(format!(
                    "'n' is too large for this SphericalExpansion: \
                    expected value below {}, got {}", self.parameters.max_radial, n
                )))
            }

            if l > self.parameters.max_angular as isize {
                return Err(Error::InvalidParameter(format!(
                    "'l' is too large for this SphericalExpansion: \
                    expected value below {}, got {}", self.parameters.max_angular + 1, l
                )))
            }

            if m < -l || m > l  {
                return Err(Error::InvalidParameter(format!(
                    "'m' is not inside [-l, l]: got m={} but l={}", m, l
                )))
            }
        }

        Ok(())
    }

    #[time_graph::instrument(name = "SphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        assert_eq!(descriptor.samples.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        self.do_self_contributions(descriptor);

        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.parameters.cutoff)?;
            let species = system.species()?;

            for pair in system.pairs()? {
                let pair = Pair {
                    system: i_system,
                    first: pair.first,
                    second: pair.second,
                    species_first: species[pair.first],
                    species_second: species[pair.second],
                    distance: pair.distance,
                    direction: pair.vector / pair.distance,
                };

                self.accumulate_for_pair(descriptor, &pair);

                if self.parameters.gradients {
                    self.accumulate_gradient_for_pair(descriptor, &pair);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::systems::test_systems;
    use crate::descriptor::{Indexes, IndexValue, IndexesBuilder};
    use crate::{Descriptor, Calculator};

    use super::{SphericalExpansion, SphericalExpansionParameters};
    use super::{CutoffFunction, RadialBasis};
    use super::super::super::CalculatorBase;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::IndexValue::from($value)
        };
    }

    fn parameters(gradients: bool) -> SphericalExpansionParameters {
        SphericalExpansionParameters {
            atomic_gaussian_width: 0.3,
            cutoff: 3.5,
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
            gradients: gradients,
            max_radial: 6,
            max_angular: 6,
            radial_basis: RadialBasis::Gto {}
        }
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(false)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]).boxed();
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems, &mut descriptor, Default::default()).unwrap();

        assert_eq!(descriptor.samples.names(), ["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.features.names(), ["n", "l", "m"]);

        let mut index = 0;
        for n in 0..6_usize {
            for l in 0..=6_isize {
                for m in -l..=l {
                    let expected = [IndexValue::from(n), IndexValue::from(l), IndexValue::from(m)];
                    assert_eq!(descriptor.features[index], expected);
                    index += 1;
                }
            }
        }

        // exact values for spherical expansion are regression-tested in
        // `rascaline/tests/spherical-expansion.rs`
    }

    #[test]
    fn finite_differences() {
        use super::super::super::tests_utils::{MovedAtomIndex, ChangedGradientIndex};

        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let system = systems.systems.pop().unwrap();

        let compute_modified_indexes = |gradients_samples: &Indexes, moved: MovedAtomIndex| {
            let mut results = Vec::new();
            for (sample_i, sample) in gradients_samples.iter().enumerate() {
                let neighbor = sample[4];
                let spatial = sample[5];
                if neighbor.usize() == moved.center && spatial.usize() == moved.spatial {
                    results.push(ChangedGradientIndex {
                        gradient_index: sample_i,
                        sample: sample[..4].to_vec(),
                    });
                }
            }
            return results;
        };

        super::super::super::tests_utils::finite_difference(
            calculator, system, compute_modified_indexes
        );
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]).boxed();

        let mut features = IndexesBuilder::new(vec!["n", "l", "m"]);
        features.add(&[v!(0), v!(1), v!(0)]);
        features.add(&[v!(3), v!(6), v!(-5)]);
        features.add(&[v!(2), v!(3), v!(2)]);
        features.add(&[v!(1), v!(4), v!(4)]);
        features.add(&[v!(5), v!(2), v!(0)]);
        features.add(&[v!(1), v!(1), v!(-1)]);
        let features = features.finish();

        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        samples.add(&[v!(0), v!(1), v!(1), v!(1)]);
        samples.add(&[v!(0), v!(2), v!(1), v!(123456)]);
        samples.add(&[v!(1), v!(0), v!(6), v!(1)]);
        samples.add(&[v!(1), v!(2), v!(1), v!(1)]);
        let samples = samples.finish();

        super::super::super::tests_utils::compute_partial(
            calculator, &mut systems, samples, features, true
        );
    }

    mod cutoff_function {
        use super::super::CutoffFunction;

        #[test]
        fn step() {
            let function = CutoffFunction::Step{};
            let cutoff = 4.0;

            assert_eq!(function.compute(2.0, cutoff), 1.0);
            assert_eq!(function.compute(5.0, cutoff), 0.0);
        }

        #[test]
        fn step_gradient() {
            let function = CutoffFunction::Step{};
            let cutoff = 4.0;

            assert_eq!(function.derivative(2.0, cutoff), 0.0);
            assert_eq!(function.derivative(5.0, cutoff), 0.0);
        }

        #[test]
        fn shifted_cosine() {
            let function = CutoffFunction::ShiftedCosine { width: 0.5 };
            let cutoff = 4.0;

            assert_eq!(function.compute(2.0, cutoff), 1.0);
            assert_eq!(function.compute(3.5, cutoff), 1.0);
            assert_eq!(function.compute(3.8, cutoff), 0.34549150281252683);
            assert_eq!(function.compute(4.0, cutoff), 0.0);
            assert_eq!(function.compute(5.0, cutoff), 0.0);
        }

        #[test]
        fn shifted_cosine_gradient() {
            let function = CutoffFunction::ShiftedCosine { width: 0.5 };
            let cutoff = 4.0;

            assert_eq!(function.derivative(2.0, cutoff), 0.0);
            assert_eq!(function.derivative(3.5, cutoff), 0.0);
            assert_eq!(function.derivative(3.8, cutoff), -2.987832164741557);
            assert_eq!(function.derivative(4.0, cutoff), 0.0);
            assert_eq!(function.derivative(5.0, cutoff), 0.0);
        }
    }
}
