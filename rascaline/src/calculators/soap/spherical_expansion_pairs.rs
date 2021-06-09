use ndarray::{Array1, Array2};

use crate::descriptor::{IndexesBuilder, IndexValue, Indexes, SamplesIndexes};
use crate::descriptor::PairSpeciesSamples;
use crate::{Descriptor, Error, System, Vector3D};

use super::super::CalculatorBase;
use super::RadialIntegral;

use super::{SphericalHarmonics, SphericalHarmonicsArray};
use super::{SphericalExpansionParameters};


/// The actual calculator used to compute SOAP spherical expansion coefficients
pub struct SphericalExpansionByPair {
    /// Parameters governing the spherical expansion
    parameters: SphericalExpansionParameters,
    /// Implementation of the radial integral
    radial_integral: Box<dyn RadialIntegral>,
    /// Implementation of the spherical harmonics
    spherical_harmonics: SphericalHarmonics,
}

impl std::fmt::Debug for SphericalExpansionByPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}

impl SphericalExpansionByPair {
    /// Create a new `SphericalExpansionByPair` calculator with the given parameters
    pub fn new(parameters: SphericalExpansionParameters) -> Result<SphericalExpansionByPair, Error> {
        Ok(SphericalExpansionByPair {
            spherical_harmonics: SphericalHarmonics::new(parameters.max_angular),
            radial_integral: parameters.radial_basis.construct(&parameters)?,
            parameters: parameters,
        })
    }
}

impl CalculatorBase for SphericalExpansionByPair {
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
        Box::new(PairSpeciesSamples::with_self_contribution(self.parameters.cutoff))
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
        assert_eq!(descriptor.samples.names(), &["structure", "first", "second", "pair_id", "species_first", "species_second"]);
        assert_eq!(descriptor.features.names(), &["n", "l", "m"]);

        let mut spherical_harmonics = SphericalHarmonicsArray::new(self.parameters.max_angular);
        let shape = (self.parameters.max_radial, self.parameters.max_angular + 1);
        let mut radial_integral = Array2::from_elem(shape, 0.0);

        // compute the self pairs first
        let mut self_contribution = Array1::from_elem(descriptor.features.count(), 0.0);
        let mut self_contribution_computed = false;
        for (sample_i, sample) in descriptor.samples.iter().enumerate() {
            let pair_id = sample[3];
            if pair_id.isize() != 0 {
                continue;
            }

            if !self_contribution_computed {
                self.radial_integral.compute(0.0, radial_integral.view_mut(), None);

                self.spherical_harmonics.compute(
                    Vector3D::new(0.0, 0.0, 1.0), &mut spherical_harmonics, None
                );
                let f_cut = self.parameters.cutoff_function.compute(0.0, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut * radial_integral[[n, l]] * spherical_harmonics[[l as isize, m]];
                    self_contribution[feature_i] = n_l_m_value;
                }

                self_contribution_computed = true;
            }

            descriptor.values.index_axis_mut(ndarray::Axis(0), sample_i).assign(&self_contribution);
        }

        // then do the actual pairs
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.parameters.cutoff)?;
            let species = system.species()?;

            for (id, pair) in system.pairs()?.iter().enumerate() {
                // TODO: this code assumes that the order of pairs do not change
                // between the creation of samples & now, this is not guaranteed
                let pair_id = (id + 1) as isize;

                let first_sample_i = descriptor.samples.position(&[
                    IndexValue::from(i_system),
                    IndexValue::from(pair.first),
                    IndexValue::from(pair.second),
                    IndexValue::from(pair_id),
                    IndexValue::from(species[pair.first]),
                    IndexValue::from(species[pair.second]),
                ]);

                let second_sample_i = descriptor.samples.position(&[
                    IndexValue::from(i_system),
                    IndexValue::from(pair.second),
                    IndexValue::from(pair.first),
                    IndexValue::from(-pair_id),
                    IndexValue::from(species[pair.second]),
                    IndexValue::from(species[pair.first]),
                ]);

                if first_sample_i.is_none() && second_sample_i.is_none() {
                    // this pair is not part of the requested samples, just
                    // continue
                    continue;
                }

                let distance = pair.distance;
                let direction = pair.vector / distance;
                self.radial_integral.compute(distance, radial_integral.view_mut(), None);
                self.spherical_harmonics.compute(
                    direction, &mut spherical_harmonics, None
                );
                let f_cut = self.parameters.cutoff_function.compute(distance, self.parameters.cutoff);

                for (feature_i, feature) in descriptor.features.iter().enumerate() {
                    let n = feature[0].usize();
                    let l = feature[1].usize();
                    let m = feature[2].isize();

                    let n_l_m_value = f_cut * radial_integral[[n, l]] * spherical_harmonics[[l as isize, m]];

                    if let Some(sample_i) = first_sample_i {
                        descriptor.values[[sample_i, feature_i]] = n_l_m_value;
                    }

                    if let Some(sample_i) = second_sample_i {
                        descriptor.values[[sample_i, feature_i]] = m_1_pow(l) * n_l_m_value;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Specialized function to compute (-1)^l. Using this instead of
/// `f64::powi(-1.0, l as i32)` decrease computation time
fn m_1_pow(l: usize) -> f64 {
    if l % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
