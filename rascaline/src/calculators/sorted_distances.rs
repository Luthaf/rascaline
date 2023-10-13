use metatensor::{Labels, LabelsBuilder, TensorMap};

use super::CalculatorBase;

use crate::{Error, System};
use crate::labels::{AtomicTypeFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{KeysBuilder, CenterTypesKeys, CenterSingleNeighborsTypesKeys};

#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// Sorted distances vector representation of an atomic environment.
///
/// Each atomic center is represented by a vector of distance to its neighbors
/// within the spherical `cutoff`, sorted from smallest to largest. If there are
/// less neighbors than `max_neighbors`, the remaining entries are filled with
/// `cutoff` instead.
///
/// Separate types for neighbors are represented separately, meaning that the
/// `max_neighbors` parameter applies one atomic type at the time.
pub struct SortedDistances {
    /// Spherical cutoff to use for atomic environments
    cutoff: f64,
    /// Maximal number of neighbors of a given atomic type a center is
    /// allowed to have. This is also the dimensionality of the features.
    max_neighbors: usize,
    /// Should separate neighbor types be represented separately?
    separate_neighbor_types: bool,
}

impl CalculatorBase for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn cutoffs(&self) -> &[f64] {
        std::slice::from_ref(&self.cutoff)
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        if self.separate_neighbor_types {
            let builder = CenterSingleNeighborsTypesKeys {
                cutoff: self.cutoff,
                self_pairs: false,
            };
            return builder.keys(systems);
        }

        return CenterTypesKeys.keys(systems);
    }

    fn sample_names(&self) -> Vec<&str> {
        AtomCenteredSamples::sample_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut samples = Vec::new();
        if self.separate_neighbor_types {
            assert_eq!(keys.names(), ["center_type", "neighbor_type"]);
            for [center_type, neighbor_type] in keys.iter_fixed_size() {
                let builder = AtomCenteredSamples {
                    cutoff: self.cutoff,
                    center_type: AtomicTypeFilter::Single(center_type.i32()),
                    neighbor_type: AtomicTypeFilter::Single(neighbor_type.i32()),
                    self_pairs: false,
                };

                samples.push(builder.samples(systems)?);
            }
        } else {
            assert_eq!(keys.names(), ["center_type"]);
            for [center_type] in keys.iter_fixed_size() {
                let builder = AtomCenteredSamples {
                    cutoff: self.cutoff,
                    center_type: AtomicTypeFilter::Single(center_type.i32()),
                    neighbor_type: AtomicTypeFilter::Any,
                    self_pairs: false,
                };

                samples.push(builder.samples(systems)?);
            }
        }

        return Ok(samples);
    }

    fn supports_gradient(&self, _parameter: &str) -> bool {
        return false;
    }

    fn positions_gradient_samples(&self, _: &Labels, _: &[Labels], _: &mut [System]) -> Result<Vec<Labels>, Error> {
        unimplemented!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        return vec![Vec::new(); keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["neighbor"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for i in 0..self.max_neighbors {
            properties.add(&[i]);
        }
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SortedDistances::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        if self.separate_neighbor_types {
            assert_eq!(descriptor.keys().names(), ["center_type", "neighbor_type"]);
        } else {
            assert_eq!(descriptor.keys().names(), ["center_type"]);
        }

        for (key, mut block) in descriptor {
            let neighbor_type = if self.separate_neighbor_types {
                Some(key[1].i32())
            } else {
                None
            };

            let block_data = block.data_mut();
            let array = block_data.values.to_array_mut();

            for (sample_i, [system_i, center_i]) in block_data.samples.iter_fixed_size().enumerate() {
                let center_i = center_i.usize();

                let system = &mut systems[system_i.usize()];
                system.compute_neighbors(self.cutoff)?;
                let types = system.types()?;

                let mut distances = Vec::new();
                for pair in system.pairs_containing(center_i)? {
                    if let Some(neighbor_type) = neighbor_type {
                        let neighbor_i = if pair.first == center_i {
                            pair.second
                        } else {
                            debug_assert_eq!(pair.second, center_i);
                            pair.first
                        };

                        if types[neighbor_i] == neighbor_type {
                            distances.push(pair.distance);
                        }
                    } else {
                        distances.push(pair.distance);
                    }
                }

                // Sort, resize to limit to at most `self.max_neighbors` values
                // and pad the distance vectors as needed
                distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                distances.resize(self.max_neighbors, self.cutoff);

                for (property_i, [neighbor]) in block_data.properties.iter_fixed_size().enumerate() {
                    array[[sample_i, property_i]] = distances[neighbor.usize()];
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{s, aview1};
    use metatensor::Labels;

    use crate::systems::test_utils::test_systems;
    use crate::Calculator;

    use super::super::CalculatorBase;
    use super::SortedDistances;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
            separate_neighbor_types: false
        }) as Box<dyn CalculatorBase>);

        assert_eq!(calculator.name(), "sorted distances vector");
        assert_eq!(calculator.parameters(), "{\"cutoff\":1.5,\"max_neighbors\":3,\"separate_neighbor_types\":false}");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SortedDistances {
            cutoff: 1.7,
            max_neighbors: 4,
            separate_neighbor_types: false
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        let keys_to_move = Labels::empty(vec!["center_type"]);
        let descriptor = descriptor.keys_to_samples(&keys_to_move, true).unwrap();

        assert_eq!(descriptor.blocks().len(), 1);
        let block = descriptor.block_by_id(0);
        let values = block.values().to_array();
        assert_eq!(values.shape(), [3, 4]);

        assert_eq!(values.slice(s![0, ..]), aview1(&[0.957897074324794, 0.957897074324794, 1.7, 1.7]));
        assert_eq!(values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.4891, 1.5109, 1.7]));
        assert_eq!(values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.4891, 1.5109, 1.7]));
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
            separate_neighbor_types: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let keys = Labels::new(["center_type"], &[[1], [6], [8], [-42]]);
        let samples = Labels::new(["system", "atom"], &[[0, 1]]);
        let properties = Labels::new(["neighbor"], &[[2], [0]]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }
}
