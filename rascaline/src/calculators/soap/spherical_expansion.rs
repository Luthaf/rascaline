use std::collections::BTreeSet;

use ndarray::Array2;

use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, SamplesIndexes, AtomSpeciesSamples};
use crate::system::Pair;
use crate::{Descriptor, System, Vector3D};

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
    fn construct(&self, parameters: &SphericalExpansionParameters) -> Box<dyn RadialIntegral> {
        match self {
            RadialBasis::Gto {} => {
                let parameters = GtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                return Box::new(GtoRadialIntegral::new(parameters));
            }
            RadialBasis::SplinedGto { accuracy } => {
                let parameters = GtoParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                let gto = GtoRadialIntegral::new(parameters);

                let parameters = SplinedRIParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    cutoff: parameters.cutoff,
                };
                return Box::new(SplinedRadialIntegral::with_accuracy(parameters, *accuracy, gto));
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

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[allow(clippy::module_name_repetitions)]
/// Parameters for spherical expansion calculator.
///
/// The spherical expansion is at the core of representations in the SOAP
/// (Smooth Overlap of Atomic Positions) family. See [this review
/// article](https://doi.org/10.1063/1.5090481) for more information on the SOAP
/// representation, and [this paper](https://doi.org/10.1063/5.0044689) for
/// information on how it is implemented in rascaline.
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

pub struct SphericalExpansion {
    parameters: SphericalExpansionParameters,
    radial_integral: Box<dyn RadialIntegral>,
    spherical_harmonics: SphericalHarmonics,
    sph_values: SphericalHarmonicsArray,
    sph_gradients: Option<[SphericalHarmonicsArray; 3]>,
    ri_values: Array2<f64>,
    ri_gradients: Option<Array2<f64>>,
}

impl SphericalExpansion {
    pub fn new(parameters: SphericalExpansionParameters) -> SphericalExpansion {
        let radial_integral = parameters.radial_basis.construct(&parameters);
        let spherical_harmonics = SphericalHarmonics::new(parameters.max_angular);
        let sph_values = SphericalHarmonicsArray::new(parameters.max_angular);
        let sph_gradients = if parameters.gradients {
            Some([
                SphericalHarmonicsArray::new(parameters.max_angular),
                SphericalHarmonicsArray::new(parameters.max_angular),
                SphericalHarmonicsArray::new(parameters.max_angular)
            ])
        } else {
            None
        };

        let shape = (parameters.max_radial, parameters.max_angular + 1);
        let ri_values = Array2::from_elem(shape, 0.0);
        let ri_gradients = if parameters.gradients {
            Some(Array2::from_elem(shape, 0.0))
        } else {
            None
        };

        SphericalExpansion {
            parameters: parameters,
            radial_integral: radial_integral,
            spherical_harmonics: spherical_harmonics,
            sph_values: sph_values,
            sph_gradients: sph_gradients,
            ri_values: ri_values,
            ri_gradients: ri_gradients,
        }
    }

    fn do_self_contributions(&mut self, descriptor: &mut Descriptor) {
        // keep a list of centers which have already been computed
        for (i_env, requested_env) in descriptor.samples.iter().enumerate() {
            let alpha = requested_env[2];
            let beta = requested_env[3];

            if alpha == beta {
                // TODO: cache self contribution, they only depend on the
                // gaussian atomic width.
                self.radial_integral.compute(0.0, self.ri_values.view_mut(), None);

                self.spherical_harmonics.compute(
                    Vector3D::new(0.0, 0.0, 1.0), &mut self.sph_values, None
                );
                let f_cut = self.parameters.cutoff_function.compute(0.0, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut * self.ri_values[[n, l]] * self.sph_values[[l as isize, m]];
                    descriptor.values[[i_env, feature_i]] += n_l_m_value;
                }
            }
        }
    }
}

impl std::fmt::Debug for SphericalExpansion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
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
        Box::new(AtomSpeciesSamples::with_self_contribution(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
        assert_eq!(indexes.names(), self.features_names());
        for value in indexes {
            let n = value[0].usize();
            let l = value[1].isize();
            let m = value[2].isize();
            assert!(n < self.parameters.max_radial);
            assert!(l <= self.parameters.max_angular as isize);
            assert!(-l <= m && m <= l);
        }
    }

    fn check_samples(&self, indexes: &Indexes, systems: &mut [Box<dyn System>]) {
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        // This could be made much faster by not recomputing the full list of
        // potential samples
        let allowed = self.samples().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid sample", value);
        }
    }

    #[allow(clippy::similar_names, clippy::too_many_lines, clippy::identity_op)]
    #[time_graph::instrument(name = "SphericalExpansion::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) {
        assert_eq!(descriptor.samples.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        self.do_self_contributions(descriptor);

        // keep the set of pairs already seen for each system
        let mut already_computed_pairs = vec![BTreeSet::new(); systems.len()];

        for (sample_i, sample) in descriptor.samples.iter().enumerate() {
            let i_system = sample[0];
            let center = sample[1].usize();
            let alpha = sample[2];
            let beta = sample[3];

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.parameters.cutoff);
            let species = system.species();

            for pair in system.pairs_containing(center) {
                let (neighbor, sign) = if center == pair.first {
                    (pair.second, 1.0)
                } else {
                    assert_eq!(center, pair.second, "system.pairs_containing is broken");
                    (pair.first, -1.0)
                };

                if species[neighbor] != beta.usize() {
                    continue;
                }

                if !already_computed_pairs[i_system.usize()].insert(sort_pair(&pair)) {
                    continue;
                }

                // we store the result for the center--neighbor pair in env_i,
                // this code check where (it can not be part of the requested
                // envs) to store the result for the neighbor--center pair.
                let other_sample_i = descriptor.samples.position(
                    &[i_system, IndexValue::from(neighbor), beta, alpha]
                );

                let distance = pair.vector.norm();
                let direction = sign * pair.vector / distance;

                self.radial_integral.compute(
                    distance, self.ri_values.view_mut(), self.ri_gradients.as_mut().map(|o| o.view_mut())
                );

                self.spherical_harmonics.compute(
                    direction, &mut self.sph_values, self.sph_gradients.as_mut()
                );
                let f_cut = self.parameters.cutoff_function.compute(distance, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut * self.ri_values[[n, l]] * self.sph_values[[l as isize, m]];
                    descriptor.values[[sample_i, feature_i]] += n_l_m_value;
                    if let Some(other_env_i) = other_sample_i {
                        // Use the fact that `se[n, l, m](-r) = (-1)^l se[n, l, m](r)`
                        // where se === spherical_expansion.
                        descriptor.values[[other_env_i, feature_i]] += m_1_pow(l) * n_l_m_value;
                    }
                }

                if self.parameters.gradients {
                    // get the indexes where to store the gradient for this
                    // specific pair, if any
                    let (center_grad_i, neighbor_grad_i) = if let Some(ref gradients_samples) = descriptor.gradients_samples {
                        assert!(self.parameters.gradients);
                        let center_grad = gradients_samples.position(&[
                            i_system, IndexValue::from(center), alpha, beta,
                            IndexValue::from(neighbor), IndexValue::from(0)
                        ]);
                        let neighbor_grad = gradients_samples.position(&[
                            i_system, IndexValue::from(neighbor), beta, alpha,
                            IndexValue::from(center), IndexValue::from(0)
                        ]);
                        (center_grad, neighbor_grad)
                    } else {
                        (None, None)
                    };

                    let f_cut_grad = self.parameters.cutoff_function.derivative(distance, self.parameters.cutoff);

                    let dr_dx = sign * pair.vector[0] / distance;
                    let dr_dy = sign * pair.vector[1] / distance;
                    let dr_dz = sign * pair.vector[2] / distance;

                    let gradients = descriptor.gradients.as_mut().expect("missing storage for gradients");
                    let center_grad_i = center_grad_i.expect("missing storage for gradient of this center");
                    let ri_gradients = self.ri_gradients.as_ref().expect("missing radial integral gradients");
                    let sph_gradients = self.sph_gradients.as_ref().expect("missing spherical harmonics gradients");

                    for (feature_i, feature) in descriptor.features.iter().enumerate() {
                        let n = feature[0].usize();
                        let l = feature[1].usize();
                        let m = feature[2].isize();

                        let sph_value = self.sph_values[[l as isize, m]];
                        let sph_grad_x = sph_gradients[0][[l as isize, m]];
                        let sph_grad_y = sph_gradients[1][[l as isize, m]];
                        let sph_grad_z = sph_gradients[2][[l as isize, m]];

                        let ri_value = self.ri_values[[n, l]];
                        let ri_grad = ri_gradients[[n, l]];

                        let grad_x = f_cut_grad * dr_dx * ri_value * sph_value
                                    + f_cut * ri_grad * dr_dx * sph_value
                                    + f_cut * ri_value * sph_grad_x / distance;

                        let grad_y = f_cut_grad * dr_dy * ri_value * sph_value
                                    + f_cut * ri_grad * dr_dy * sph_value
                                    + f_cut * ri_value * sph_grad_y / distance;

                        let grad_z = f_cut_grad * dr_dz * ri_value * sph_value
                                    + f_cut * ri_grad * dr_dz * sph_value
                                    + f_cut * ri_value * sph_grad_z / distance;

                        // assumes that the three spatial derivative are stored
                        // one after the other
                        gradients[[center_grad_i + 0, feature_i]] += grad_x;
                        gradients[[center_grad_i + 1, feature_i]] += grad_y;
                        gradients[[center_grad_i + 2, feature_i]] += grad_z;

                        if let Some(neighbor_grad_i) = neighbor_grad_i {
                            // Use the fact that `grad se[n, l, m](-r) = (-1)^(l + 1) grad se[n, l, m](r)`
                            // where se === spherical_expansion.
                            let parity = m_1_pow(l + 1);
                            gradients[[neighbor_grad_i + 0, feature_i]] = parity * grad_x;
                            gradients[[neighbor_grad_i + 1, feature_i]] = parity * grad_y;
                            gradients[[neighbor_grad_i + 2, feature_i]] = parity * grad_z;
                        }
                    }
                }
            }
        }
    }
}

fn sort_pair(pair: &Pair) -> (usize, usize) {
    if pair.first <= pair.second {
        (pair.first, pair.second)
    } else {
        (pair.second, pair.first)
    }
}

#[cfg(test)]
mod tests {
    use crate::system::test_systems;
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
        )) as Box<dyn CalculatorBase>);

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
        )) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let system = systems.systems.pop().unwrap();

        let compute_modified_indexes = |gradients_samples: &Indexes, moved: MovedAtomIndex| {
            let mut results = Vec::new();
            for (sample_i, sample) in gradients_samples.iter().enumerate() {
                let center = sample[1];
                let neighbor = sample[4];
                let spatial = sample[5];
                if center.usize() != moved.center &&
                   neighbor.usize() == moved.center &&
                   spatial.usize() == moved.spatial
                {
                    results.push(ChangedGradientIndex {
                        gradient_index: sample_i,
                        sample: sample[..4].to_vec(),
                    });
                }
            }
            return results;
        };

        let max_relative = 5e-4;
        super::super::super::tests_utils::finite_difference(
            calculator, system, compute_modified_indexes, max_relative
        );
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansion::new(
            parameters(true)
        )) as Box<dyn CalculatorBase>);

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
