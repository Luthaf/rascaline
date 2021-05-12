use std::collections::HashMap;

use ndarray::{aview1, s};

use super::CalculatorBase;

use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
use crate::descriptor::{SamplesIndexes, AtomSpeciesSamples};
use crate::{Descriptor, Error, System};

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// Sorted distances vector representation of an atomic environment.
///
/// Each atomic center is represented by a vector of distance to its neighbors
/// within the spherical `cutoff`, sorted from smallest to largest. If there are
/// less neighbors than `max_neighbors`, the remaining entries are filled with
/// `cutoff` instead.
///
/// Separate species for neighbors are represented separately, meaning that the
/// `max_neighbors` parameter only apply to a single species.
pub struct SortedDistances {
    /// Spherical cutoff to use for atomic environments
    cutoff: f64,
    /// Maximal number of neighbors of a given atomic species a center is
    /// allowed to have. This is also the dimensionality of the features.
    max_neighbors: usize,
}

impl CalculatorBase for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn get_parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn features_names(&self) -> Vec<&str> {
        vec!["neighbor"]
    }

    fn features(&self) -> Indexes {
        let mut features = IndexesBuilder::new(self.features_names());
        for i in 0..self.max_neighbors {
            features.add(&[IndexValue::from(i)]);
        }
        return features.finish();
    }

    fn samples(&self) -> Box<dyn SamplesIndexes> {
        Box::new(AtomSpeciesSamples::new(self.cutoff))
    }

    fn compute_gradients(&self) -> bool {
        false
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        assert_eq!(indexes.names(), self.features_names());
        for value in indexes.iter() {
            if value[0].usize() >= self.max_neighbors {
                return Err(Error::InvalidParameter(format!(
                    "neighbor index is too large for this SortedDistances: \
                    got {}, expected value lower than {}", value[0].usize(), self.max_neighbors
                )))
            }
        }
        Ok(())
    }

    #[time_graph::instrument(name = "SortedDistances::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        let all_features = descriptor.features.count() == self.max_neighbors;
        let mut requested_features = Vec::new();
        if !all_features {
            for feature in descriptor.features.iter() {
                let neighbor = feature[0];
                requested_features.push(neighbor);
            }
        }

        // index of the first entry of descriptor.values corresponding to
        // the current system
        let mut current = 0;
        for (i_system, system) in systems.iter_mut().enumerate() {
            // distance contains a vector of distances vector (one distance
            // vector for each center) for each pair of species in the system
            let mut distances = HashMap::new();
            for sample in &descriptor.samples {
                let alpha = sample[2].usize();
                let beta = sample[3].usize();
                distances.entry((alpha, beta)).or_insert_with(
                    || vec![Vec::with_capacity(self.max_neighbors); system.size()]
                );
            }

            // Collect all distances around each center in `distances`
            system.compute_neighbors(self.cutoff);
            let species = system.species();
            for pair in system.pairs() {
                let i = pair.first;
                let j = pair.second;
                let d = pair.distance;

                if let Some(distances) = distances.get_mut(&(species[i], species[j])) {
                    distances[i].push(d);
                }

                if let Some(distances) = distances.get_mut(&(species[j], species[i])) {
                    distances[j].push(d);
                }
            }

            // Sort, resize to limit to at most `self.max_neighbors` values
            // and pad the distance vectors as needed
            for vectors in distances.iter_mut().map(|(_, vectors)| vectors) {
                for vec in vectors {
                    vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    vec.resize(self.max_neighbors, self.cutoff);
                }
            }

            loop {
                if current == descriptor.samples.count() {
                    break;
                }

                // Copy the data in the descriptor array, until we find the
                // next system
                if let [structure, center, alpha, beta] = descriptor.samples[current] {
                    if structure.usize() != i_system {
                        break;
                    }

                    let distance_vector = &distances.get(&(alpha.usize(), beta.usize())).unwrap()[center.usize()];
                    if all_features {
                        descriptor.values.slice_mut(s![current, ..]).assign(&aview1(distance_vector));
                    } else {
                        // Only assign the requested values
                        for (i, &neighbor) in requested_features.iter().enumerate() {
                            descriptor.values[[current, i]] = distance_vector[neighbor.usize()];
                        }
                    }
                } else {
                    unreachable!();
                }
                current += 1;
            }
        }

        // sanity check: did we get all samples in the above loop?
        assert_eq!(current, descriptor.samples.count());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::systems::test_systems;
    use crate::{Descriptor, Calculator};
    use crate::{CalculationOptions, SelectedIndexes};
    use crate::descriptor::{IndexesBuilder, IndexValue};

    use super::super::CalculatorBase;

    use ndarray::{s, aview1};

    use super::SortedDistances;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
        }) as Box<dyn CalculatorBase>);

        assert_eq!(calculator.name(), "sorted distances vector");
        assert_eq!(calculator.parameters(), "{\"cutoff\":1.5,\"max_neighbors\":3}");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]).boxed();
        let mut descriptor = Descriptor::new();
        calculator.compute(&mut systems, &mut descriptor, Default::default()).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 3]);

        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
    }

    #[test]
    #[ignore]
    fn gradients() {
        unimplemented!()
    }

    #[test]
    fn compute_partial() {
        let mut calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]).boxed();
        let mut descriptor = Descriptor::new();

        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        samples.add(&[
            IndexValue::from(0_usize), IndexValue::from(1),
            IndexValue::from(1_usize), IndexValue::from(123456)
        ]);
        let options = CalculationOptions {
            selected_samples: SelectedIndexes::Some(samples.finish()),
            selected_features: SelectedIndexes::All,
            ..Default::default()
        };
        calculator.compute(&mut systems, &mut descriptor, options).unwrap();

        assert_eq!(descriptor.values.shape(), [1, 3]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));

        let mut features = IndexesBuilder::new(vec!["neighbor"]);
        features.add(&[IndexValue::from(0)]);
        features.add(&[IndexValue::from(2)]);

        let options = CalculationOptions {
            selected_samples: SelectedIndexes::All,
            selected_features: SelectedIndexes::Some(features.finish()),
            ..Default::default()
        };
        calculator.compute(&mut systems, &mut descriptor, options).unwrap();

        assert_eq!(descriptor.values.shape(), [3, 2]);
        assert_eq!(descriptor.values.slice(s![0, ..]), aview1(&[0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5]));
        assert_eq!(descriptor.values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5]));
    }
}
