use std::collections::BTreeSet;

use ndarray::Array2;

use crate::descriptor::Descriptor;
use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, EnvironmentIndexes, AtomSpeciesEnvironment};
use crate::system::System;

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
        Box::new(AtomSpeciesEnvironment::new(self.parameters.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        self.parameters.gradients
    }

    fn check_features(&self, indexes: &Indexes) {
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
        assert_eq!(indexes.names(), &["structure", "center", "alpha", "beta"]);
        // This could be made much faster by not recomputing the full list of
        // potential environments
        let allowed = self.environments().indexes(systems);
        for value in indexes.iter() {
            assert!(allowed.contains(value), "{:?} is not a valid environment", value);
        }
    }

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn compute(&mut self, systems: &mut [&mut dyn System], descriptor: &mut Descriptor) {
        assert_eq!(descriptor.environments.names(), &["structure", "center", "alpha", "beta"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        for (i_env, requested_env) in descriptor.environments.iter().enumerate() {
            let i_system = requested_env[0];
            let center = requested_env[1].usize();
            let alpha = requested_env[2];
            let beta = requested_env[3];

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.parameters.cutoff);

            // keep a list of pairs for which everything have already been
            // computed
            let mut already_computed_pairs = BTreeSet::new();

            // TODO: add a system.pairs_with(center) function instead of
            // searching through all pairs at all time
            for pair in system.pairs() {
                if (pair.first == center || pair.second == center) && already_computed_pairs.insert((pair.first, pair.second)) {
                    let distance = pair.vector.norm();
                    let direction = pair.vector / distance;

                    let (other_env_i, sign) = if center == pair.first {
                        (descriptor.environments.position(&[i_system, IndexValue::from(pair.second), beta, alpha]), 1.0)
                    } else {
                        (descriptor.environments.position(&[i_system, IndexValue::from(pair.first), beta, alpha]), -1.0)
                    };

                    self.radial_integral.compute(
                        distance, self.ri_values.view_mut(), self.ri_gradients.as_mut().map(|o| o.view_mut())
                    );

                    self.spherical_harmonics.compute(
                        sign * direction, &mut self.sph_values, self.sph_gradients.as_mut()
                    );

                    let f_cut = self.parameters.cutoff_function.compute(distance, self.parameters.cutoff);

                    for (i_feature, feature) in descriptor.features.iter().enumerate() {
                        let n = feature[0].usize();
                        let l = feature[1].usize();
                        let m = feature[1].isize();

                        let n_l_m_value = f_cut * self.ri_values[[n, l]] * self.sph_values[[l as isize, m]];
                        descriptor.values[[i_env, i_feature]] += sign.powi(l as i32) * n_l_m_value;
                        if let Some(other_env_i) = other_env_i {
                            descriptor.values[[other_env_i, i_feature]] += (-sign).powi(l as i32) * n_l_m_value;
                        }
                    }

                    // get the indexes where to store the gradient for this
                    // specific pair, if any
                    let (center_grad_i, neighbor_grad_i) = if let Some(ref indexes) = descriptor.gradients_indexes {
                        assert!(self.parameters.gradients);
                        if center == pair.first {
                            let center_grad = indexes.position(&[
                                i_system, IndexValue::from(pair.first), alpha, beta,
                                IndexValue::from(pair.second), IndexValue::from(0_usize)
                            ]);
                            let neighbor_grad = indexes.position(&[
                                i_system, IndexValue::from(pair.second), beta, alpha,
                                IndexValue::from(pair.first), IndexValue::from(0_usize)
                            ]);
                            (center_grad, neighbor_grad)
                        } else {
                            let center_grad = indexes.position(&[
                                i_system, IndexValue::from(pair.second), alpha, beta,
                                IndexValue::from(pair.first), IndexValue::from(0_usize)
                            ]);
                            let neighbor_grad = indexes.position(&[
                                i_system, IndexValue::from(pair.first), beta, alpha,
                                IndexValue::from(pair.second), IndexValue::from(0_usize)
                            ]);
                            (center_grad, neighbor_grad)
                        }
                    } else {
                        (None, None)
                    };

                    if self.parameters.gradients {
                        let f_cut_grad = self.parameters.cutoff_function.derivative(distance, self.parameters.cutoff);

                        let dr_dx = pair.vector[0] / distance;
                        let dr_dy = pair.vector[1] / distance;
                        let dr_dz = pair.vector[2] / distance;

                        let gradients = descriptor.gradients.as_mut().expect("missing storage for gradients");
                        let center_grad_i = center_grad_i.expect("missing storage for gradient");
                        let neighbor_grad_i = neighbor_grad_i.expect("missing storage for gradient");
                        let ri_gradients = self.ri_gradients.as_ref().expect("missing radial integral gradients");
                        let sph_gradients = self.sph_gradients.as_ref().expect("missing spherical harmonics gradients");

                        for (i_feature, feature) in descriptor.features.iter().enumerate() {
                            let n = feature[0].usize();
                            let l = feature[1].usize();
                            let m = feature[1].isize();

                            let sph_value = self.sph_values[[l as isize, m]];
                            let sph_grad_x = sph_gradients[0][[l as isize, m]];
                            let sph_grad_y = sph_gradients[1][[l as isize, m]];
                            let sph_grad_z = sph_gradients[2][[l as isize, m]];

                            let ri_value = self.ri_values[[n, l]];
                            let ri_grad = ri_gradients[[n, l]];

                            let grad_x = f_cut_grad * dr_dx * ri_value * sph_value
                                       + f_cut * ri_grad * dr_dx * sph_value
                                       + f_cut * ri_value * sph_grad_x;

                            let grad_y = f_cut_grad * dr_dy * ri_value * sph_value
                                       + f_cut * ri_grad * dr_dy * sph_value
                                       + f_cut * ri_value * sph_grad_y;

                            let grad_z = f_cut_grad * dr_dz * ri_value * sph_value
                                       + f_cut * ri_grad * dr_dz * sph_value
                                       + f_cut * ri_value * sph_grad_z;

                            // assumes that the three spatial derivative are one after the other
                            gradients[[center_grad_i, i_feature]] += sign.powi(l as i32) * grad_x;
                            gradients[[neighbor_grad_i, i_feature]] += (-sign).powi(l as i32) * grad_x;

                            gradients[[center_grad_i + 1, i_feature]] += sign.powi(l as i32) * grad_y;
                            gradients[[neighbor_grad_i + 1, i_feature]] += (-sign).powi(l as i32) * grad_y;

                            gradients[[center_grad_i + 2, i_feature]] += sign.powi(l as i32) * grad_z;
                            gradients[[neighbor_grad_i + 2, i_feature]] += (-sign).powi(l as i32) * grad_z;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn tests() {
        todo!()
    }
}
