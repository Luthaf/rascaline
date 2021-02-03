use std::collections::BTreeSet;

use ndarray::Array2;

use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, EnvironmentIndexes, AtomSpeciesEnvironment};
use crate::system::Pair;
use crate::{Descriptor, System, Vector3D};

use super::super::CalculatorBase;
use super::{GTO, GTOParameters, RadialIntegral};
use super::{SphericalHarmonics, SphericalHarmonicsArray};

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize)]
pub enum RadialBasis {
    GTO,
}

/// Possible values for the smoothing cutoff function
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum CutoffFunction {
    /// Step function, 1 if `r < r_cut` and 0 if `r >= r_cut`
    Step,
    /// Shifted cosine switching function
    /// f(r) = 1/2 * (1 + cos(\pi (r - r_cut + width) / width ))
    ShiftedCosine {
        width: f64,
    },
}

impl CutoffFunction {
    /// Evaluate the cutoff function at the distance `r` for the given `cutoff`
    pub fn compute(&self, r: f64, cutoff: f64) -> f64 {
        match self {
            CutoffFunction::Step => {
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
            CutoffFunction::Step => 0.0,
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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[allow(clippy::module_name_repetitions)]
pub struct SphericalExpansionParameters {
    /// Spherical cutoff to use for atomic environments
    pub cutoff: f64,
    /// Number of radial basis function to use
    pub max_radial: usize,
    /// Number of spherical harmonics to use
    pub max_angular: usize,
    /// Width of the atom-centered gaussian creating the atomic density
    pub atomic_gaussian_width: f64,
    /// Should we also compute gradients of the feature?
    pub gradients: bool,
    /// radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// cutoff function used to smooth the behavior around the cutoff radius
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
        let radial_integral = match parameters.radial_basis {
            RadialBasis::GTO => {
                let parameters = GTOParameters {
                    max_radial: parameters.max_radial,
                    max_angular: parameters.max_angular,
                    atomic_gaussian_width: parameters.atomic_gaussian_width,
                    cutoff: parameters.cutoff,
                };
                Box::new(GTO::new(parameters))
            }
        };

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
        for (i_env, requested_env) in descriptor.environments.iter().enumerate() {
            let alpha = requested_env[2];
            let beta = requested_env[3];

            if alpha == beta {
                // TODO: cache self contribution, they only depend on the
                // gaussian atomic width
                self.radial_integral.compute(0.0, self.ri_values.view_mut(), None);

                self.spherical_harmonics.compute(
                    Vector3D::new(0.0, 0.0, 1.0), &mut self.sph_values, None
                );
                let f_cut = self.parameters.cutoff_function.compute(0.0, self.parameters.cutoff);

                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[1].isize();

                    let n_l_m_value = f_cut * self.ri_values[[n, l]] * self.sph_values[[l as isize, m]];
                    descriptor.values[[i_env, i_feature]] += n_l_m_value;
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

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(vec!["n", "l", "m"]);
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

    fn environments(&self) -> Box<dyn EnvironmentIndexes> {
        Box::new(AtomSpeciesEnvironment::with_self_contribution(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
        // TODO check for duplicated features?
        assert_eq!(indexes.names(), &["n", "l", "m"]);
        for value in indexes {
            let n = value[0].usize();
            let l = value[0].isize();
            let m = value[0].isize();
            assert!(n < self.parameters.max_radial);
            assert!(l <= self.parameters.max_angular as isize);
            assert!(-l <= m && m <= l);
        }
    }

    fn check_environments(&self, indexes: &Indexes, systems: &mut [&mut dyn System]) {
        // TODO: check for duplicated environments?
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        // This could be made much faster by not recomputing the full list of
        // potential environments
        let allowed = self.environments().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid environment", value);
        }
    }

    #[allow(clippy::similar_names, clippy::too_many_lines, clippy::identity_op)]
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        assert_eq!(descriptor.environments.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        self.do_self_contributions(descriptor);

        // keep the set of pairs already seen for each system
        let mut already_computed_pairs = vec![BTreeSet::new(); systems.len()];

        for (i_env, requested_env) in descriptor.environments.iter().enumerate() {
            let i_system = requested_env[0];
            let center = requested_env[1].usize();
            let alpha = requested_env[2];
            let beta = requested_env[3];

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.parameters.cutoff);
            let species = system.species();

            // TODO: add a system.pairs_with(center) function instead of
            // searching through all pairs at all time
            for pair in system.pairs() {
                if pair.first != center && pair.second != center {
                    continue;
                }

                let (neighbor, sign) = if center == pair.first {
                    (pair.second, 1.0)
                } else {
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
                let other_env_i = descriptor.environments.position(
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

                for (i_feature, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut * self.ri_values[[n, l]] * self.sph_values[[l as isize, m]];
                    descriptor.values[[i_env, i_feature]] += n_l_m_value;
                    if let Some(other_env_i) = other_env_i {
                        // Use the fact that `se[n, l, m](-r) = (-1)^l se[n, l, m](r)`
                        // where se === spherical_expansion.
                        descriptor.values[[other_env_i, i_feature]] += f64::powi(-1.0, l as i32) * n_l_m_value;
                    }
                }

                if self.parameters.gradients {
                    // get the indexes where to store the gradient for this
                    // specific pair, if any
                    let (center_grad_i, neighbor_grad_i) = if let Some(ref gradients_indexes) = descriptor.gradients_indexes {
                        assert!(self.parameters.gradients);
                        let center_grad = gradients_indexes.position(&[
                            i_system, IndexValue::from(center), alpha, beta,
                            IndexValue::from(neighbor), IndexValue::from(0_usize)
                        ]);
                        let neighbor_grad = gradients_indexes.position(&[
                            i_system, IndexValue::from(neighbor), beta, alpha,
                            IndexValue::from(center), IndexValue::from(0_usize)
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
                    let center_grad_i = center_grad_i.expect("missing storage for gradient");
                    let ri_gradients = self.ri_gradients.as_ref().expect("missing radial integral gradients");
                    let sph_gradients = self.sph_gradients.as_ref().expect("missing spherical harmonics gradients");

                    for (i_feature, feature) in descriptor.features.iter().enumerate() {
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
                        gradients[[center_grad_i + 0, i_feature]] += grad_x;
                        gradients[[center_grad_i + 1, i_feature]] += grad_y;
                        gradients[[center_grad_i + 2, i_feature]] += grad_z;

                        if let Some(neighbor_grad_i) = neighbor_grad_i {
                            // Use the fact that `grad se[n, l, m](-r) = (-1)^(l + 1) grad se[n, l, m](r)`
                            // where se === spherical_expansion.
                            let parity = f64::powi(-1.0, l as i32 + 1);
                            gradients[[neighbor_grad_i + 0, i_feature]] = parity * grad_x;
                            gradients[[neighbor_grad_i + 1, i_feature]] = parity * grad_y;
                            gradients[[neighbor_grad_i + 2, i_feature]] = parity * grad_z;
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
    use crate::descriptor::{IndexValue, IndexesBuilder};
    use crate::{Descriptor, Calculator, System};

    use approx::assert_relative_eq;
    use ndarray::s;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::IndexValue::from($value as f64)
        };
    }

    fn hyperparameters(gradients: bool) -> String {
        format!("{{
            \"atomic_gaussian_width\": 0.3,
            \"cutoff\": 3.5,
            \"cutoff_function\": {{
              \"ShiftedCosine\": {{
                \"width\": 0.5
              }}
            }},
            \"gradients\": {},
            \"max_radial\": 6,
            \"max_angular\": 6,
            \"radial_basis\": \"GTO\"
        }}", gradients)
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::new(
            "spherical_expansion",
            hyperparameters(false)
        ).unwrap();

        let mut systems = test_systems(&["water"]);
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut descriptor);

        assert_eq!(descriptor.environments.names(), ["structure", "center", "species_center", "species_neighbor"]);
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
        let mut calculator = Calculator::new(
            "spherical_expansion",
            hyperparameters(true)
        ).unwrap();

        let mut systems = test_systems(&["water"]);
        let mut reference = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut reference);

        // exact gradients for spherical expansion are regression-tested in
        // `rascaline/tests/spherical-expansion.rs`

        let gradients_indexes = reference.gradients_indexes.as_ref().unwrap();
        assert_eq!(
            gradients_indexes.names(),
            ["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]
        );

        // get the list of modified gradient environments when moving atom_i
        let modified_indexes = |atom_i: usize, spatial_index: usize| {
            let mut results = Vec::new();
            for (env_i, env) in gradients_indexes.iter().enumerate() {
                let center = env[1];
                let neighbor = env[4];
                let spatial = env[5];
                if center.usize() != atom_i && neighbor.usize() == atom_i && spatial.usize() == spatial_index {
                    results.push((env_i, &env[..4]));
                }
            }
            return results;
        };

        let delta = 1e-9;
        let gradients = reference.gradients.as_ref().unwrap();
        for atom_i in 0..systems.systems[0].size() {
            for spatial in 0..3 {
                systems.systems[0].positions_mut()[atom_i][spatial] += delta;

                let mut updated = Descriptor::new();
                calculator.compute(&mut systems.get(), &mut updated);

                for (grad_i, env) in modified_indexes(atom_i, spatial) {
                    let env_i = reference.environments.position(env).expect(
                        "missing environment in reference values"
                    );
                    assert_eq!(updated.environments.position(env).unwrap(), env_i);

                    let value = reference.values.slice(s![env_i, ..]);
                    let value_delta = updated.values.slice(s![env_i, ..]);
                    let gradient = gradients.slice(s![grad_i, ..]);

                    assert_eq!(value.shape(), value_delta.shape());
                    assert_eq!(value.shape(), gradient.shape());

                    let mut finite_difference = value_delta.to_owned().clone();
                    finite_difference -= &value;
                    finite_difference /= delta;

                    assert_relative_eq!(
                        finite_difference, gradient,
                        epsilon=1e-9,
                        max_relative=5e-4,
                    );
                }

                systems.systems[0].positions_mut()[atom_i][spatial] -= delta;
            }
        }
    }

    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::new(
            "spherical_expansion",
            hyperparameters(true),
        ).unwrap();

        let mut systems = test_systems(&["water", "methane"]);
        let mut full = Descriptor::new();
        calculator.compute(&mut systems.get(), &mut full);

        // all features, all environments
        let mut partial = Descriptor::new();
        calculator.compute_partial(&mut systems.get(), &mut partial, None, None);

        assert_eq!(full.environments, partial.environments);
        assert_eq!(full.features, partial.features);
        assert_eq!(full.values, partial.values);
        assert_eq!(full.gradients_indexes, partial.gradients_indexes);
        assert_eq!(full.gradients, partial.gradients);

        // partial set of features, all environments
        let mut features = IndexesBuilder::new(vec!["n", "l", "m"]);
        features.add(&[v!(0), v!(1), v!(0)]);
        features.add(&[v!(3), v!(6), v!(-5)]);
        features.add(&[v!(2), v!(3), v!(2)]);
        features.add(&[v!(1), v!(4), v!(4)]);
        features.add(&[v!(5), v!(2), v!(0)]);
        features.add(&[v!(1), v!(1), v!(-1)]);
        let features = features.finish();
        calculator.compute_partial(&mut systems.get(), &mut partial, None, Some(features.clone()));

        assert_eq!(full.environments, partial.environments);
        for (partial_i, feature) in features.iter().enumerate() {
            let index = full.features.position(feature).unwrap();
            assert_eq!(
                full.values.slice(s![.., index]),
                partial.values.slice(s![.., partial_i])
            );

            assert_eq!(
                full.gradients.as_ref().unwrap().slice(s![.., index]),
                partial.gradients.as_ref().unwrap().slice(s![.., partial_i])
            );
        }

        // all features, partial set of environments
        let mut environments = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        environments.add(&[v!(0), v!(1), v!(1), v!(1)]);
        environments.add(&[v!(0), v!(2), v!(1), v!(123456)]);
        environments.add(&[v!(1), v!(0), v!(6), v!(1)]);
        environments.add(&[v!(1), v!(2), v!(1), v!(1)]);
        let environments = environments.finish();
        calculator.compute_partial(&mut systems.get(), &mut partial, Some(environments.clone()), None);

        assert_eq!(full.features, partial.features);
        for (partial_i, environment) in environments.iter().enumerate() {
            let index = full.environments.position(environment).unwrap();
            assert_eq!(
                full.values.slice(s![index, ..]),
                partial.values.slice(s![partial_i, ..])
            );

        }
        for (partial_i, environment) in partial.gradients_indexes.as_ref().unwrap().iter().enumerate() {
            let index = full.gradients_indexes.as_ref().unwrap().position(environment).unwrap();
            assert_eq!(
                full.gradients.as_ref().unwrap().slice(s![index, ..]),
                partial.gradients.as_ref().unwrap().slice(s![partial_i, ..])
            );
        }

        // partial set of features, partial set of environments
        calculator.compute_partial(&mut systems.get(), &mut partial, Some(environments.clone()), Some(features.clone()));
        for (env_i, environment) in environments.iter().enumerate() {
            for (feature_i, feature) in features.iter().enumerate() {
                let full_env = full.environments.position(environment).unwrap();
                let full_feature = full.features.position(feature).unwrap();
                assert_eq!(
                    full.values[[full_env, full_feature]],
                    partial.values[[env_i, feature_i]]
                );
            }
        }

        for (env_i, environment) in partial.gradients_indexes.as_ref().unwrap().iter().enumerate() {
            for (feature_i, feature) in features.iter().enumerate() {
                let full_env = full.gradients_indexes.as_ref().unwrap().position(environment).unwrap();
                let full_feature = full.features.position(feature).unwrap();
                assert_eq!(
                    full.gradients.as_ref().unwrap()[[full_env, full_feature]],
                    partial.gradients.as_ref().unwrap()[[env_i, feature_i]]
                );
            }
        }
    }

    mod cutoff_function {
        use super::super::CutoffFunction;

        #[test]
        fn step() {
            let function = CutoffFunction::Step;
            let cutoff = 4.0;

            assert_eq!(function.compute(2.0, cutoff), 1.0);
            assert_eq!(function.compute(5.0, cutoff), 0.0);
        }

        #[test]
        fn step_gradient() {
            let function = CutoffFunction::Step;
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
