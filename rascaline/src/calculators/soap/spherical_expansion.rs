use std::cell::RefCell;

use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis};
use thread_local::ThreadLocal;

use crossbeam::channel::Sender;

use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, SamplesBuilder, TwoBodiesSpeciesSamples};
use crate::{Descriptor, Error, System, Vector3D};
use crate::types::StackVec;

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
        let code = parameters.radial_basis.construct(parameters)?;
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

/// Identify which atom in a pair we are referring to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomInPair {
    First,
    Second,
}

impl Default for AtomInPair {
    fn default() -> Self {
        AtomInPair::First
    }
}

/// Contribution of a single pair to the spherical expansion. This will be
/// created in a "compute" thread and send for accumulation in the main values
/// array to a "writer" thread.
struct PairContribution {
    /// A PairContribution can contribute to up to 2 samples.
    ///
    /// For each sample, the first element in the tuple is the position in the
    /// spherical expansion array where we want to accumulate the values (i.e.
    /// the index of the center); and the second element in the tuple describe
    /// whether the center is the first or second atom in the pair.
    ///
    /// The use of `StackVec` instead of `Vec` or `SmallVec` is a performance
    /// optimization.
    samples: StackVec<[(usize, AtomInPair); 2]>,
    /// pair contribution to the spherical expansion
    values: Array1<f64>,
}

impl PairContribution {
    /// Create a new pair contribution with the given feature `size`
    fn new(size: usize) -> PairContribution {
        PairContribution {
            samples: StackVec::new(),
            values: Array1::from_elem(size, 0.0),
        }
    }

    /// Add a sample to which this pair contribution should be accumulated into
    fn add_sample(&mut self, index: usize, center_position_in_pair: AtomInPair) {
        self.samples.push((index, center_position_in_pair));
    }
}

/// Contribution of a single pair to the spherical expansion gradients. This
/// will be created in a "compute" thread and send for accumulation in the main
/// values array to a "writer" thread.
struct GradientsPairContribution {
    /// A pair can contribute to up to 4 gradients entries.
    ///
    /// For each sample, the first element in the tuple is the position in the
    /// gradient array of the x component of the gradient. The second and third
    /// elements in the tuple describe whether the center/neighbor atoms are the
    /// the first or second atom in the pair.
    ///
    /// The use of `StackVec` instead of `Vec` or `SmallVec` is a performance
    /// optimization.
    samples: StackVec<[(usize, AtomInPair, AtomInPair); 4]>,
    /// gradient w.r.t. each of the cartesian coordinate
    gradients: [Array1<f64>; 3],
}

impl GradientsPairContribution {
    /// Create a new gradient pair contribution with the given feature `size`
    fn new(size: usize) -> GradientsPairContribution {
        GradientsPairContribution {
            samples: StackVec::new(),
            gradients: [
                Array1::from_elem(size, 0.0),
                Array1::from_elem(size, 0.0),
                Array1::from_elem(size, 0.0),
            ],
        }
    }

    /// Add a sample to which this pair contribution should be accumulated into
    fn add_sample(&mut self, index: usize, center_position_in_pair: AtomInPair, neighbor_position_in_pair: AtomInPair) {
        self.samples.push((index, center_position_in_pair, neighbor_position_in_pair));
    }
}

/// The actual calculator used to compute SOAP spherical expansion coefficients
pub struct SphericalExpansion {
    /// Parameters governing the spherical expansion
    parameters: SphericalExpansionParameters,
    radial_integral: ThreadLocal<RefCell<RadialIntegralImpl>>,
    spherical_harmonics: ThreadLocal<RefCell<SphericalHarmonicsImpl>>,
}

impl std::fmt::Debug for SphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SphericalExpansion {
    /// Create a new `SphericalExpansion` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansion, Error> {
        // validate parameters once in the constructor, so that we can use
        // expect("invalid parameters") in the main code.
        RadialIntegralImpl::new(&parameters)?;

        return Ok(SphericalExpansion {
            parameters,
            radial_integral: ThreadLocal::new(),
            spherical_harmonics: ThreadLocal::new(),
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

        let mut radial_integral = self.radial_integral.get_or(|| {
            let ri = RadialIntegralImpl::new(&self.parameters).expect("invalid parameters");
            RefCell::new(ri)
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsImpl::new(&self.parameters))
        }).borrow_mut();

        for (i_env, requested_env) in descriptor.samples.iter().enumerate() {
            let species_center = requested_env[2];
            let species_neighbor = requested_env[3];

            if species_center == species_neighbor {
                radial_integral.compute_no_gradients(0.0);
                spherical_harmonics.compute_no_gradients(Vector3D::new(0.0, 0.0, 1.0));
                let f_cut = self.parameters.cutoff_function.compute(0.0, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut
                        * radial_integral.values[[n, l]]
                        * spherical_harmonics.values[[l as isize, m]];
                    descriptor.values[[i_env, feature_i]] += n_l_m_value;
                }
            }
        }
    }

    /// Accumulate the spherical expansion coefficients for the given pair.
    ///
    /// This function passes results back to calling code though `sender`
    fn accumulate_for_pair(
        &self,
        sender: &Sender<PairContribution>,
        samples: &Indexes,
        features: &Indexes,
        pair: &Pair
    ) {
        let first_sample_i = samples.position(&[
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
            second_sample_i = samples.position(&[
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

        let mut radial_integral = self.radial_integral.get_or(|| {
            let ri = RadialIntegralImpl::new(&self.parameters).expect("invalid parameters");
            RefCell::new(ri)
        }).borrow_mut();

        let mut spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsImpl::new(&self.parameters))
        }).borrow_mut();

        radial_integral.compute(pair.distance);
        spherical_harmonics.compute(pair.direction);

        let f_cut = self.parameters.cutoff_function.compute(
            pair.distance, self.parameters.cutoff
        );

        let mut pair_contribution = PairContribution::new(features.count());
        if let Some(index) = first_sample_i {
            pair_contribution.add_sample(index, AtomInPair::First);
        }

        if let Some(index) = second_sample_i {
            pair_contribution.add_sample(index, AtomInPair::Second);
        }

        for (feature_i, feature) in features.iter().enumerate() {
            let n = feature[0].usize();
            let l = feature[1].usize();
            let m = feature[2].isize();

            let n_l_m_value = f_cut
                * radial_integral.values[[n, l]]
                * spherical_harmonics.values[[l as isize, m]];

            pair_contribution.values[feature_i] = n_l_m_value;
        }

        sender.send(pair_contribution).expect("receiver hanged up");
    }

    /// Accumulate the spherical expansion gradients for the given pair.
    ///
    /// This function assumes that the radial integral and spherical harmonics
    /// have just been computed for this pair.
    #[allow(clippy::needless_range_loop)]
    fn accumulate_gradient_for_pair(
        &self,
        sender: &Sender<GradientsPairContribution>,
        gradients_samples: &Indexes,
        features: &Indexes,
        pair: &Pair
    ) {
        debug_assert!(self.parameters.gradients);

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

        let mut pair_contribution = GradientsPairContribution::new(features.count());
        if let Some(index) = first_grad_i {
            pair_contribution.add_sample(index, AtomInPair::First, AtomInPair::Second);
        }

        if let Some(index) = first_self_grad_i {
            pair_contribution.add_sample(index, AtomInPair::First, AtomInPair::First);
        }

        if let Some(index) = second_grad_i {
            pair_contribution.add_sample(index, AtomInPair::Second, AtomInPair::First);
        }

        if let Some(index) = second_self_grad_i {
            pair_contribution.add_sample(index, AtomInPair::Second, AtomInPair::Second);
        }

        let radial_integral = self.radial_integral.get_or(|| {
            let ri = RadialIntegralImpl::new(&self.parameters).expect("invalid parameters");
            RefCell::new(ri)
        }).borrow();

        let spherical_harmonics = self.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsImpl::new(&self.parameters))
        }).borrow();

        let ri_values = &radial_integral.values;
        let ri_gradients = radial_integral.gradients.as_ref().expect("missing radial integral gradients");

        let sph_values = &spherical_harmonics.values;
        let sph_gradients = spherical_harmonics.gradients.as_ref().expect("missing spherical harmonics gradients");

        let f_cut = self.parameters.cutoff_function.compute(
            pair.distance, self.parameters.cutoff
        );
        let f_cut_grad = self.parameters.cutoff_function.derivative(
            pair.distance, self.parameters.cutoff
        );

        for spatial in 0..3 {
            let dr_d_spatial = pair.direction[spatial];
            let sph_gradient = &sph_gradients[spatial];
            let gradients = &mut pair_contribution.gradients[spatial];

            for (feature_i, feature) in features.iter().enumerate() {
                let n = feature[0].usize();
                let l = feature[1].usize();
                let m = feature[2].isize();

                let sph_value = sph_values[[l as isize, m]];
                let sph_grad = sph_gradient[[l as isize, m]];

                let ri_value = ri_values[[n, l]];
                let ri_grad = ri_gradients[[n, l]];

                gradients[feature_i] = f_cut_grad * dr_d_spatial * ri_value * sph_value
                                     + f_cut * ri_grad * dr_d_spatial * sph_value
                                     + f_cut * ri_value * sph_grad / pair.distance;
            }
        }

        sender.send(pair_contribution).expect("receiver hanged up");
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
    species_first: i32,
    /// species of the second atom of the pair
    species_second: i32,
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

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
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
    #[allow(clippy::enum_glob_use)]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        assert_eq!(descriptor.samples.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        self.do_self_contributions(descriptor);

        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.parameters.cutoff)?;
            let species = system.species()?;

            let pairs = system.pairs()?;

            // Setup parallel computation.
            //
            // This code distribute work for computing the spherical expansion
            // over multiple threads, iterating over pairs in parallel.
            //
            // To workaround the fact that each working thread would like to
            // write data to potentially the same location as other threads
            // (since two different pairs can produce data to be stored in the
            // same environment), we use channels to send data two other threads
            // (one for values, one for gradients) that only receive data and
            // write it to the descriptor.
            let samples = &descriptor.samples;
            let gradient_samples = descriptor.gradients_samples.as_ref();
            let features = &descriptor.features;
            let values = &mut descriptor.values;
            let gradients = descriptor.gradients.as_mut();

            // use crossbeam channels instead of std::sync::mpsc::SyncChannel
            // since crossbeam is faster in our case.
            let (sender_values, receiver_values) = crossbeam::channel::unbounded::<PairContribution>();
            let (sender_grad, receiver_grad) = crossbeam::channel::unbounded::<GradientsPairContribution>();

            // use crossbeam scoped threads instead of rayon's, to ensure we
            // make progress even with RAYON_NUM_THREADS=1
            crossbeam::thread::scope(|s|{
                // re-borrow self as an immutable reference to be passed to
                // the first closure below
                let this = &*self;

                // Start a thread to produce values. This thread will start
                // RAYON_NUM_THREADS more threads, actually doing the work.
                s.spawn(move |_| {
                    pairs.par_iter()
                        .map(|pair| (pair, sender_values.clone(), sender_grad.clone()))
                        .for_each(|(pair, sender_values, sender_grad)| {
                            let mut pair = Pair {
                                system: i_system,
                                first: pair.first,
                                second: pair.second,
                                species_first: species[pair.first],
                                species_second: species[pair.second],
                                distance: pair.distance,
                                direction: pair.vector / pair.distance,
                            };

                            // Deal with the possibility that two atoms are at
                            // the same position. While this is not usual, there
                            // is no reason to prevent the calculation of
                            // spherical expansion. The user will still get a
                            // warning about atoms being very close together
                            // when calculating the neighbor list.
                            if pair.distance < 1e-6 {
                                pair.direction = Vector3D::new(0.0, 0.0, 1.0);
                            }

                            this.accumulate_for_pair(
                                &sender_values,
                                samples,
                                features,
                                &pair
                            );

                            if this.parameters.gradients {
                                this.accumulate_gradient_for_pair(
                                    &sender_grad,
                                    gradient_samples.expect("missing gradient samples"),
                                    features,
                                    &pair,
                                );
                            }
                        });
                });

                // Start a thread to receive and collect values
                s.spawn(move |_| {
                    let m_1_pow_l = features.iter()
                        .map(|feature| m_1_pow(feature[1].usize()))
                        .collect::<Array1<f64>>();

                    for contribution in receiver_values {
                        for &(index, center) in contribution.samples.iter() {
                            let mut row = values.index_axis_mut(Axis(0), index);
                            match center {
                                AtomInPair::First => {
                                    row += &contribution.values;
                                }
                                AtomInPair::Second => {
                                    // Use the fact that `se[n, l, m](-r) =
                                    // (-1)^l se[n, l, m](r)` where se is the
                                    // spherical expansion
                                    row += &(m_1_pow_l.clone() * &contribution.values);
                                }
                            }
                        }

                    }
                });

                // Start a thread to receive and collect gradients
                if self.parameters.gradients {
                    let gradients = gradients.expect("missing storage for gradients");
                    s.spawn(move |_| {
                        use self::AtomInPair::*;

                        let m_1_pow_l = features.iter()
                            .map(|feature| m_1_pow(feature[1].usize()))
                            .collect::<Array1<f64>>();

                        for contribution in receiver_grad {
                            for &(index, center, neighbor) in contribution.samples.iter() {
                                for spatial in 0..3 {
                                    let gradient = &contribution.gradients[spatial];
                                    // we assume that the three spatial
                                    // components are stored one after the other
                                    let mut row = gradients.index_axis_mut(Axis(0), index + spatial);

                                    match (center, neighbor) {
                                        (First, Second) => {
                                            row += gradient;
                                        }
                                        (First, First) => {
                                            row -= gradient;
                                        }
                                        // when storing data for "reversed"
                                        // gradients, use the fact that `grad_j
                                        // se_i[n, l, m](r) = - (-1)^l grad_i
                                        // se_j[n, l, m](r)` where se is the
                                        // spherical expansion.
                                        (Second, First) => {
                                            row -= &(m_1_pow_l.clone() * gradient);
                                        }
                                        (Second, Second) => {
                                            row += &(m_1_pow_l.clone() * gradient);
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }).expect("one of the thread panicked");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::systems::test_utils::{test_systems, test_system};
    use crate::descriptor::{IndexValue, IndexesBuilder};
    use crate::{Descriptor, Calculator};

    use super::{SphericalExpansion, SphericalExpansionParameters};
    use super::{CutoffFunction, RadialBasis};
    use crate::calculators::CalculatorBase;

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

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

        let mut systems = test_systems(&["water"]);
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
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let system = test_system("water");
        crate::calculators::tests_utils::finite_difference(calculator, system);
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(true)
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let mut features = IndexesBuilder::new(vec!["n", "l", "m"]);
        features.add(&[v(0), v(1), v(0)]);
        features.add(&[v(3), v(6), v(-5)]);
        features.add(&[v(2), v(3), v(2)]);
        features.add(&[v(1), v(4), v(4)]);
        features.add(&[v(5), v(2), v(0)]);
        features.add(&[v(1), v(1), v(-1)]);
        let features = features.finish();

        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        samples.add(&[v(0), v(1), v(1), v(1)]);
        samples.add(&[v(0), v(2), v(1), v(123456)]);
        samples.add(&[v(1), v(0), v(6), v(1)]);
        samples.add(&[v(1), v(2), v(1), v(1)]);
        let samples = samples.finish();

        crate::calculators::tests_utils::compute_partial(
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
