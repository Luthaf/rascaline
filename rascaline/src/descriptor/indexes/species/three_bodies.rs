use std::collections::BTreeSet;

use itertools::Itertools;

use crate::{Error, System};
use crate::systems::Pair;
use super::super::{SamplesBuilder, Indexes, IndexesBuilder, IndexValue};

/// `ThreeBodiesSpeciesSamples` is used to represents atom-centered environments
/// representing three body atomic density correlation; where the three bodies
/// include the central atom and two neighbors. These environments include
/// chemical species information.
///
/// The base set of indexes contains `structure`, `center` (i.e. central atom
/// index inside the structure), `species_center`, `species_neighbor_1` and
/// `species_neighbor2`; the gradient indexes also contains the `neighbor`
/// inside the spherical cutoff with respect to which the gradient is taken and
/// the `spatial` (i.e x/y/z) index.
pub struct ThreeBodiesSpeciesSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
    /// Is the central atom considered to be its own neighbor?
    self_contribution: bool,
}

impl ThreeBodiesSpeciesSamples {
    /// Create a new `ThreeBodiesSpeciesSamples` with the given `cutoff`, excluding
    /// self contributions.
    pub fn new(cutoff: f64) -> ThreeBodiesSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for ThreeBodiesSpeciesSamples");
        ThreeBodiesSpeciesSamples {
            cutoff: cutoff,
            self_contribution: false,
        }
    }

    /// Create a new `ThreeBodiesSpeciesSamples` with the given `cutoff`,
    /// including self contributions.
    pub fn with_self_contribution(cutoff: f64) -> ThreeBodiesSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for ThreeBodiesSpeciesSamples");
        ThreeBodiesSpeciesSamples {
            cutoff: cutoff,
            self_contribution: true,
        }
    }
}

/// A Set built as a sorted vector
struct SortedVecSet<T> {
    data: Vec<T>
}

impl<T: Ord> SortedVecSet<T> {
    fn new() -> Self {
        SortedVecSet {
            data: Vec::new()
        }
    }

    fn insert(&mut self, value: T) {
        match self.data.binary_search(&value) {
            Ok(_) => {},
            Err(index) => self.data.insert(index, value),
        }
    }
}

impl<T> IntoIterator for SortedVecSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl SamplesBuilder for ThreeBodiesSpeciesSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        // Accumulate indexes in a set first to ensure uniqueness of the indexes
        // even if their are multiple neighbors of the same specie around a
        // given center
        let mut set = SortedVecSet::new();

        let sort_pair = |i, j| {
            if i < j { (i, j) } else { (j, i) }
        };
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            for center in 0..system.size()? {
                let pairs = system.pairs_containing(center)?;
                for (i, j) in triplets_from_pairs(pairs, center) {
                    let (species_1, species_2) = sort_pair(species[i], species[j]);
                    set.insert((i_system, center, species[center], species_1, species_2));
                }
            }

            if self.self_contribution {
                for (center, &species_center) in species.iter().enumerate() {
                    set.insert((i_system, center, species_center, species_center, species_center));

                    for pair in system.pairs_containing(center)? {
                        let neighbor = if pair.first == center {
                            pair.second
                        } else {
                            pair.first
                        };

                        let (species_1, species_2) = sort_pair(species_center, species[neighbor]);
                        set.insert((i_system, center, species_center, species_1, species_2));
                    }
                }
            }
        }

        let mut indexes = IndexesBuilder::new(self.names());
        for (structure, center, species_center, species_1, species_2) in set {
            indexes.add(&[
                IndexValue::from(structure),
                IndexValue::from(center),
                IndexValue::from(species_center),
                IndexValue::from(species_1),
                IndexValue::from(species_2)
            ]);
        }

        return Ok(indexes.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) ->Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        let sort_pair = |i, j| {
            if i < j { (i, j) } else { (j, i) }
        };

        let mut indexes = BTreeSet::new();
        for (i_sample, sample) in samples.iter().enumerate() {
            let i_system = sample[0];
            let center = sample[1].usize();
            let species_center = sample[2].i32();
            let species_neighbor_1 = sample[3].i32();
            let species_neighbor_2 = sample[4].i32();

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff)?;

            let species = system.species()?;

            if self.self_contribution {
                let species_neighbor = if species_neighbor_1 == species_center {
                    Some(species_neighbor_2)
                } else if species_neighbor_2 == species_center {
                    Some(species_neighbor_1)
                } else {
                    None
                };

                if let Some(species_neighbor) = species_neighbor {
                    // include the gradient of an environnement w.r.t its center
                    // even if there is no neighbor of the other type around.
                    indexes.insert((i_sample, center));

                    for pair in system.pairs_containing(center)? {
                        let neighbor = if pair.first == center {
                            pair.second
                        } else {
                            assert_eq!(pair.second, center);
                            pair.first
                        };

                        if species[neighbor] != species_neighbor {
                            continue;
                        }

                        indexes.insert((i_sample, neighbor));
                    }
                }
            }

            let pairs = system.pairs_containing(center)?;
            for (i, j) in triplets_from_pairs(pairs, center) {
                let (species_1, species_2) = sort_pair(species[i], species[j]);

                if species_1 == species_neighbor_1 && species_2 == species_neighbor_2 {
                    indexes.insert((i_sample, center));
                    indexes.insert((i_sample, i));
                    indexes.insert((i_sample, j));
                }
            }
        }

        let mut gradients = IndexesBuilder::new(vec!["sample", "atom", "spatial"]);
        for (i_sample, atom) in indexes {
            gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(0)]);
            gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(1)]);
            gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(2)]);
        }

        return Ok(Some(gradients.finish()));
    }
}

/// Build the list of triplet i-center-j from the given list of pairs
fn triplets_from_pairs(pairs: &[Pair], center: usize) -> impl Iterator<Item=(usize, usize)> + '_ {
    pairs.iter().cartesian_product(pairs).map(move |(first_pair, second_pair)| {
        let i = if first_pair.first == center {
            first_pair.second
        } else {
            first_pair.first
        };

        let j = if second_pair.first == center {
            second_pair.second
        } else {
            second_pair.first
        };

        return (i, j);
    })
}


#[cfg(test)]
#[allow(clippy::identity_op)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    #[test]
    fn three_bodies() {
        let mut systems = test_systems(&["water"]);
        let builder = ThreeBodiesSpeciesSamples::new(2.0);

        let (samples, gradients) = builder.with_gradients(&mut systems).unwrap();
        assert_eq!(samples.count(), 7);
        assert_eq!(samples.names(), &["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            // H-H channel around O in water
            &[v(0), v(0), v(123456), v(1), v(1)],
            // H-H channel around first H in water
            &[v(0), v(1), v(1), v(1), v(1)],
            // H-O channel around first H in water
            &[v(0), v(1), v(1), v(1), v(123456)],
            // O-O channel around first H in water
            &[v(0), v(1), v(1), v(123456), v(123456)],
            // H-H channel around second H in water
            &[v(0), v(2), v(1), v(1), v(1)],
            // H-O channel around second H in water
            &[v(0), v(2), v(1), v(1), v(123456)],
            // O-O channel around second H in water
            &[v(0), v(2), v(1), v(123456), v(123456)],
        ]);

        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 51);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-H channel around O, derivative w.r.t. central O
            [v(0), v(0), v(0)], [v(0), v(0), v(1)], [v(0), v(0), v(2)],
            // H-H channel around O, derivative w.r.t. H1
            [v(0), v(1), v(0)], [v(0), v(1), v(1)], [v(0), v(1), v(2)],
            // H-H channel around O, derivative w.r.t. H2
            [v(0), v(2), v(0)],[v(0), v(2), v(1)],[v(0), v(2), v(2)],
            // H-H channel around H1, derivative w.r.t. H1
            [v(1), v(1), v(0)], [v(1), v(1), v(1)], [v(1), v(1), v(2)],
            // H-H channel around H1, derivative w.r.t. H2
            [v(1), v(2), v(0)], [v(1), v(2), v(1)], [v(1), v(2), v(2)],
            // H-O channel around H1, derivative w.r.t. O
            [v(2), v(0), v(0)], [v(2), v(0), v(1)], [v(2), v(0), v(2)],
            // H-O channel around H1, derivative w.r.t. H1
            [v(2), v(1), v(0)], [v(2), v(1), v(1)], [v(2), v(1), v(2)],
            // H-O channel around H1, derivative w.r.t. H2
            [v(2), v(2), v(0)], [v(2), v(2), v(1)], [v(2), v(2), v(2)],
            // O-O channel around H1, derivative w.r.t. O
            [v(3), v(0), v(0)], [v(3), v(0), v(1)], [v(3), v(0), v(2)],
            // O-O channel around H1, derivative w.r.t. H1
            [v(3), v(1), v(0)], [v(3), v(1), v(1)], [v(3), v(1), v(2)],
            // H-H channel around H2, derivative w.r.t. H1
            [v(4), v(1), v(0)], [v(4), v(1), v(1)], [v(4), v(1), v(2)],
            // H-H channel around H2, derivative w.r.t. H2
            [v(4), v(2), v(0)], [v(4), v(2), v(1)], [v(4), v(2), v(2)],
            // H-O channel around H2, derivative w.r.t. O
            [v(5), v(0), v(0)], [v(5), v(0), v(1)], [v(5), v(0), v(2)],
            // H-O channel around H2, derivative w.r.t. H1
            [v(5), v(1), v(0)], [v(5), v(1), v(1)], [v(5), v(1), v(2)],
            // H-O channel around H2, derivative w.r.t. H2
            [v(5), v(2), v(0)], [v(5), v(2), v(1)], [v(5), v(2), v(2)],
            // O-O channel around H2, derivative w.r.t. O
            [v(6), v(0), v(0)], [v(6), v(0), v(1)], [v(6), v(0), v(2)],
            // O-O channel around H2, derivative w.r.t. H2
            [v(6), v(2), v(0)], [v(6), v(2), v(1)], [v(6), v(2), v(2)],
        ]);

    }

    #[test]
    fn three_bodies_partial_gradients() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        samples.add(&[v(0), v(1), v(6), v(1), v(1)]);
        samples.add(&[v(1), v(1), v(1), v(123456), v(123456)]);
        samples.add(&[v(0), v(0), v(1), v(6), v(6)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["CH", "water"]);
        let builder = ThreeBodiesSpeciesSamples::new(2.0);

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 18);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-H channel around C, derivative w.r.t. H
            [v(0), v(0), v(0)], [v(0), v(0), v(1)], [v(0), v(0), v(2)],
            // H-H channel around C, derivative w.r.t. C
            [v(0), v(1), v(0)], [v(0), v(1), v(1)], [v(0), v(1), v(2)],
            // O-O channel around H1, derivative w.r.t. O
            [v(1), v(0), v(0)], [v(1), v(0), v(1)], [v(1), v(0), v(2)],
            // O-O channel around H1, derivative w.r.t. H1
            [v(1), v(1), v(0)], [v(1), v(1), v(1)], [v(1), v(1), v(2)],
            // C-C channel around H, derivative w.r.t. H
            [v(2), v(0), v(0)], [v(2), v(0), v(1)], [v(2), v(0), v(2)],
            // C-C channel around H, derivative w.r.t. C
            [v(2), v(1), v(0)], [v(2), v(1), v(1)], [v(2), v(1), v(2)],
        ]);
    }

    #[test]
    fn self_contribution() {
        let mut systems = test_systems(&["water"]);
        // Only include O-H neighbors with this cutoff
        let builder = ThreeBodiesSpeciesSamples::with_self_contribution(1.2);
        assert_eq!(builder.names(), &["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);

        let (samples, gradients) = builder.with_gradients(&mut systems).unwrap();
        assert_eq!(samples.names(), builder.names());
        assert_eq!(samples.count(), 9);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            // H-H channel around O
            &[v(0), v(0), v(123456), v(1), v(1)],
            // O-H channel around O
            &[v(0), v(0), v(123456), v(1), v(123456)],
            // O-O channel around O
            &[v(0), v(0), v(123456), v(123456), v(123456)],
            // H-H channel around H1
            &[v(0), v(1), v(1), v(1), v(1)],
            // O-H channel around H1
            &[v(0), v(1), v(1), v(1), v(123456)],
            // O-O channel around H1
            &[v(0), v(1), v(1), v(123456), v(123456)],
            // H-H channel around H2
            &[v(0), v(2), v(1), v(1), v(1)],
            // O-H channel around H2
            &[v(0), v(2), v(1), v(1), v(123456)],
            // O-O channel around H2
            &[v(0), v(2), v(1), v(123456), v(123456)],
        ]);


        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 51);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-H channel around O, gradients w.r.t. O, H1, H2
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            // O-H channel around O, gradients w.r.t. O, H1, H2
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            &[v(1), v(2), v(0)], &[v(1), v(2), v(1)], &[v(1), v(2), v(2)],
            // O-O channel around O, gradients w.r.t. O
            &[v(2), v(0), v(0)], &[v(2), v(0), v(1)], &[v(2), v(0), v(2)],
            // H-H channel around H1, gradients w.r.t. H1
            &[v(3), v(1), v(0)], &[v(3), v(1), v(1)], &[v(3), v(1), v(2)],
            // O-H channel around H1, gradients w.r.t. O, H1
            &[v(4), v(0), v(0)], &[v(4), v(0), v(1)], &[v(4), v(0), v(2)],
            &[v(4), v(1), v(0)], &[v(4), v(1), v(1)], &[v(4), v(1), v(2)],
            // O-O channel around H1, gradients w.r.t. O, H1
            &[v(5), v(0), v(0)], &[v(5), v(0), v(1)], &[v(5), v(0), v(2)],
            &[v(5), v(1), v(0)], &[v(5), v(1), v(1)], &[v(5), v(1), v(2)],
            // H-H channel around H2, gradients w.r.t. H2
            &[v(6), v(2), v(0)], &[v(6), v(2), v(1)], &[v(6), v(2), v(2)],
            // O-H channel around H2, gradients w.r.t. O, H2
            &[v(7), v(0), v(0)], &[v(7), v(0), v(1)], &[v(7), v(0), v(2)],
            &[v(7), v(2), v(0)], &[v(7), v(2), v(1)], &[v(7), v(2), v(2)],
            // O-O channel around H2, gradients w.r.t. O, H2
            &[v(8), v(0), v(0)], &[v(8), v(0), v(1)], &[v(8), v(0), v(2)],
            &[v(8), v(2), v(0)], &[v(8), v(2), v(1)], &[v(8), v(2), v(2)],
        ]);
    }

    #[test]
    fn self_contribution_partial_gradients() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor_1", "species_neighbor_2"]);
        samples.add(&[v(0), v(1), v(6), v(1), v(1)]);
        samples.add(&[v(1), v(1), v(1), v(123456), v(123456)]);
        samples.add(&[v(0), v(0), v(1), v(1), v(6)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["CH", "water"]);
        let builder = ThreeBodiesSpeciesSamples::with_self_contribution(2.0);

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 18);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H-H around C in CH, gradients w.r.t. C, H
            [v(0), v(0), v(0)], [v(0), v(0), v(1)], [v(0), v(0), v(2)],
            [v(0), v(1), v(0)], [v(0), v(1), v(1)], [v(0), v(1), v(2)],
            // O-O around H1 in water, gradients w.r.t. O, H1
            [v(1), v(0), v(0)], [v(1), v(0), v(1)], [v(1), v(0), v(2)],
            [v(1), v(1), v(0)], [v(1), v(1), v(1)], [v(1), v(1), v(2)],
            // C-H around H in CH, gradients w.r.t. C, H
            [v(2), v(0), v(0)], [v(2), v(0), v(1)], [v(2), v(0), v(2)],
            [v(2), v(1), v(0)], [v(2), v(1), v(1)], [v(2), v(1), v(2)],
        ]);
    }
}
