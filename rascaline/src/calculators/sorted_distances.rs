use std::sync::Arc;

use equistore::{Labels, LabelsBuilder, LabelValue, TensorMap};

use super::CalculatorBase;

use crate::{Error, System};
use crate::labels::{SpeciesFilter, SamplesBuilder};
use crate::labels::AtomCenteredSamples;
use crate::labels::{KeysBuilder, CenterSpeciesKeys, CenterSingleNeighborsSpeciesKeys};

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
    /// Should separate neighbor species be represented separately?
    separate_neighbor_species: bool,
}

impl CalculatorBase for SortedDistances {
    fn name(&self) -> String {
        "sorted distances vector".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        if self.separate_neighbor_species {
            let builder = CenterSingleNeighborsSpeciesKeys {
                cutoff: self.cutoff,
                self_pairs: false,
            };
            return builder.keys(systems);
        }

        return CenterSpeciesKeys.keys(systems);
    }

    fn samples_names(&self) -> Vec<&str> {
        AtomCenteredSamples::samples_names()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        let mut samples = Vec::new();
        if self.separate_neighbor_species {
            assert_eq!(keys.names(), ["species_center", "species_neighbor"]);
            for [species_center, species_neighbor] in keys.iter_fixed_size() {
                let builder = AtomCenteredSamples {
                    cutoff: self.cutoff,
                    species_center: SpeciesFilter::Single(species_center.i32()),
                    species_neighbor: SpeciesFilter::Single(species_neighbor.i32()),
                    self_pairs: false,
                };

                samples.push(builder.samples(systems)?);
            }
        } else {
            assert_eq!(keys.names(), ["species_center"]);
            for [species_center] in keys.iter_fixed_size() {
                let builder = AtomCenteredSamples {
                    cutoff: self.cutoff,
                    species_center: SpeciesFilter::Single(species_center.i32()),
                    species_neighbor: SpeciesFilter::Any,
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

    fn positions_gradient_samples(&self, _: &Labels, _: &[Arc<Labels>], _: &mut [Box<dyn System>]) -> Result<Vec<Arc<Labels>>, Error> {
        unimplemented!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Arc<Labels>>> {
        return vec![Vec::new(); keys.count()];
    }

    fn properties_names(&self) -> Vec<&str> {
        vec!["neighbor"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Arc<Labels>> {
        let mut properties = LabelsBuilder::new(self.properties_names());
        for i in 0..self.max_neighbors {
            properties.add(&[LabelValue::from(i)]);
        }
        let properties = Arc::new(properties.finish());

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "SortedDistances::compute")]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        if self.separate_neighbor_species {
            assert_eq!(descriptor.keys().names(), ["species_center", "species_neighbor"]);
        } else {
            assert_eq!(descriptor.keys().names(), ["species_center"]);
        }

        for (key, mut block) in descriptor.iter_mut() {
            let species_neighbor = if self.separate_neighbor_species {
                Some(key[1].i32())
            } else {
                None
            };

            let values = block.values_mut();
            let array = values.data.as_array_mut();

            for (sample_i, [structure_i, center_i]) in values.samples.iter_fixed_size().enumerate() {
                let center_i = center_i.usize();

                let system = &mut systems[structure_i.usize()];
                system.compute_neighbors(self.cutoff)?;
                let species = system.species()?;

                let mut distances = Vec::new();
                for pair in system.pairs_containing(center_i)? {
                    if let Some(species_neighbor) = species_neighbor {
                        let neighbor_i = if pair.first == center_i {
                            pair.second
                        } else {
                            debug_assert_eq!(pair.second, center_i);
                            pair.first
                        };

                        if species[neighbor_i] == species_neighbor {
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

                for (property_i, [neighbor]) in values.properties.iter_fixed_size().enumerate() {
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
    use equistore::{LabelsBuilder, LabelValue};

    use crate::systems::test_utils::test_systems;
    use crate::Calculator;

    use super::super::CalculatorBase;
    use super::SortedDistances;

    #[test]
    fn name_and_parameters() {
        let calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
            separate_neighbor_species: false
        }) as Box<dyn CalculatorBase>);

        assert_eq!(calculator.name(), "sorted distances vector");
        assert_eq!(calculator.parameters(), "{\"cutoff\":1.5,\"max_neighbors\":3,\"separate_neighbor_species\":false}");
    }

    #[test]
    fn values() {
        let mut calculator = Calculator::from(Box::new(SortedDistances {
            cutoff: 1.5,
            max_neighbors: 3,
            separate_neighbor_species: false
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);
        let mut descriptor = calculator.compute(&mut systems, Default::default()).unwrap();
        let keys_to_move = LabelsBuilder::new(vec!["species_center"]).finish();
        descriptor.keys_to_samples(&keys_to_move, true).unwrap();

        assert_eq!(descriptor.blocks().len(), 1);
        let values = descriptor.blocks()[0].values().data.as_array();
        assert_eq!(values.shape(), [3, 3]);

        assert_eq!(values.slice(s![0, ..]), aview1(&[0.957897074324794, 0.957897074324794, 1.5]));
        assert_eq!(values.slice(s![1, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
        assert_eq!(values.slice(s![2, ..]), aview1(&[0.957897074324794, 1.5, 1.5]));
    }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SortedDistances{
            cutoff: 1.5,
            max_neighbors: 3,
            separate_neighbor_species: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let mut samples = LabelsBuilder::new(vec!["structure", "center"]);
        samples.add(&[LabelValue::new(0), LabelValue::new(1)]);

        let mut properties = LabelsBuilder::new(vec!["neighbor"]);
        properties.add(&[LabelValue::new(2)]);
        properties.add(&[LabelValue::new(0)]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &samples.finish(), &properties.finish()
        );
    }
}
