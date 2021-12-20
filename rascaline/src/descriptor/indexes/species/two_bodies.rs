use std::collections::BTreeSet;

use crate::{Error, System};
use super::super::{SamplesBuilder, Indexes, IndexesBuilder, IndexValue};

/// `TwoBodiesSpeciesSamples` is used to represents atom-centered environments,
/// where each atom in a structure is described with a feature vector based on
/// other atoms inside a sphere centered on the central atom. These environments
/// include chemical species information.
///
/// The base set of indexes contains `structure`, `center` (i.e. central atom
/// index inside the structure), `species_center` and `species_neighbor`; the
/// gradient indexes also contains the `neighbor` inside the spherical cutoff
/// with respect to which the gradient is taken and the `spatial` (i.e x/y/z)
/// index.
pub struct TwoBodiesSpeciesSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
    /// Is the central atom considered to be its own neighbor?
    self_contribution: bool,
}

impl TwoBodiesSpeciesSamples {
    /// Create a new `TwoBodiesSpeciesSamples` with the given `cutoff`, excluding
    /// self contributions.
    pub fn new(cutoff: f64) -> TwoBodiesSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for TwoBodiesSpeciesSamples");
        TwoBodiesSpeciesSamples {
            cutoff: cutoff,
            self_contribution: false,
        }
    }

    /// Create a new `TwoBodiesSpeciesSamples` with the given `cutoff`, including
    /// self contributions.
    pub fn with_self_contribution(cutoff: f64) -> TwoBodiesSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for TwoBodiesSpeciesSamples");
        TwoBodiesSpeciesSamples {
            cutoff: cutoff,
            self_contribution: true,
        }
    }
}

impl SamplesBuilder for TwoBodiesSpeciesSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center", "species_center", "species_neighbor"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        // Accumulate indexes in a set first to ensure uniqueness of the indexes
        // even if their are multiple neighbors of the same specie around a
        // given center
        let mut set = BTreeSet::new();
        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            if self.self_contribution {
                for (center, &species) in species.iter().enumerate() {
                    set.insert((i_system, center, species, species));
                }
            }

            for pair in system.pairs()? {
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                set.insert((i_system, pair.first, species_first, species_second));
                set.insert((i_system, pair.second, species_second, species_first));
            };
        }

        let mut indexes = IndexesBuilder::new(self.names());
        for (s, c, a, b) in set {
            indexes.add(&[
                IndexValue::from(s), IndexValue::from(c), IndexValue::from(a), IndexValue::from(b)
            ]);
        }

        return Ok(indexes.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        let mut indexes = BTreeSet::new();
        for (i_sample, sample) in samples.iter().enumerate() {
            let i_system = sample[0];
            let center = sample[1].usize();
            let species_center = sample[2].i32();
            let species_neighbor = sample[3].i32();

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff)?;

            if species_neighbor == species_center && self.self_contribution {
                indexes.insert((i_sample, center));
            }

            let species = system.species()?;
            for pair in system.pairs_containing(center)? {
                let neighbor = if pair.first == center { pair.second } else { pair.first };

                if species[neighbor] != species_neighbor {
                    continue;
                }

                indexes.insert((i_sample, center));
                indexes.insert((i_sample, neighbor));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    #[test]
    fn samples() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = TwoBodiesSpeciesSamples::new(2.0);
        assert_eq!(builder.names(), &["structure", "center", "species_center", "species_neighbor"]);

        let (samples, gradients) = builder.with_gradients(&mut systems).unwrap();
        assert_eq!(samples.names(), builder.names());
        assert_eq!(samples.count(), 7);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            // C channel around H in CH
            &[v(0), v(0), v(1), v(6)],
            // H channel around C in CH
            &[v(0), v(1), v(6), v(1)],
            // H channel around O in water
            &[v(1), v(0), v(123456), v(1)],
            // H channel around the first H in water
            &[v(1), v(1), v(1), v(1)],
            // O channel around the first H in water
            &[v(1), v(1), v(1), v(123456)],
            // H channel around the second H in water
            &[v(1), v(2), v(1), v(1)],
            // O channel around the second H in water
            &[v(1), v(2), v(1), v(123456)],
        ]);

        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 45);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // C channel around H, derivative w.r.t. H
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            // C channel around H, derivative w.r.t. C
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            // H channel around C, derivative w.r.t. H
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            // H channel around C, derivative w.r.t. C
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            // H channel around O in water
            &[v(2), v(0), v(0)], &[v(2), v(0), v(1)], &[v(2), v(0), v(2)],
            &[v(2), v(1), v(0)], &[v(2), v(1), v(1)], &[v(2), v(1), v(2)],
            &[v(2), v(2), v(0)], &[v(2), v(2), v(1)], &[v(2), v(2), v(2)],
            // H channel around the H1 in water
            &[v(3), v(1), v(0)], &[v(3), v(1), v(1)], &[v(3), v(1), v(2)],
            &[v(3), v(2), v(0)], &[v(3), v(2), v(1)], &[v(3), v(2), v(2)],
            // O channel around the H1 in water
            &[v(4), v(0), v(0)], &[v(4), v(0), v(1)], &[v(4), v(0), v(2)],
            &[v(4), v(1), v(0)], &[v(4), v(1), v(1)], &[v(4), v(1), v(2)],
            // H channel around the H2 in water
            &[v(5), v(1), v(0)], &[v(5), v(1), v(1)], &[v(5), v(1), v(2)],
            &[v(5), v(2), v(0)], &[v(5), v(2), v(1)], &[v(5), v(2), v(2)],
            // O channel around the H2 in water
            &[v(6), v(0), v(0)], &[v(6), v(0), v(1)], &[v(6), v(0), v(2)],
            &[v(6), v(2), v(0)], &[v(6), v(2), v(1)], &[v(6), v(2), v(2)],
        ]);
    }

    #[test]
    fn partial_gradients() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        samples.add(&[v(1), v(0), v(123456), v(1)]);
        samples.add(&[v(0), v(0), v(1), v(6)]);
        samples.add(&[v(1), v(1), v(1), v(1)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["CH", "water"]);
        let builder = TwoBodiesSpeciesSamples::new(2.0);
        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 21);
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around O in water, derivative w.r.t. O, H1, H2
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            // C channel around H in CH, derivative w.r.t. H, C
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            // H channel around H1 in water, derivative w.r.t. H1, H2
            &[v(2), v(1), v(0)], &[v(2), v(1), v(1)], &[v(2), v(1), v(2)],
            &[v(2), v(2), v(0)], &[v(2), v(2), v(1)], &[v(2), v(2), v(2)]
        ]);
    }

    #[test]
    fn self_contribution() {
        let mut systems = test_systems(&["CH"]);
        let builder = TwoBodiesSpeciesSamples::with_self_contribution(2.0);
        assert_eq!(builder.names(), &["structure", "center", "species_center", "species_neighbor"]);

        let (samples, gradients) = builder.with_gradients(&mut systems).unwrap();
        assert_eq!(samples.names(), builder.names());
        assert_eq!(samples.count(), 4);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            // H channel around H in CH
            &[v(0), v(0), v(1), v(1)],
            // C channel around H in CH
            &[v(0), v(0), v(1), v(6)],
            // H channel around C in CH
            &[v(0), v(1), v(6), v(1)],
            // C channel around C in CH
            &[v(0), v(1), v(6), v(6)],
        ]);

        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 18);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around H atom, gradient w.r.t. H positions
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            // C channel around H atom, gradient w.r.t. H positions
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            // C channel around H atom, gradient w.r.t. C positions
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            // H channel around C atom, gradient w.r.t. H positions
            &[v(2), v(0), v(0)], &[v(2), v(0), v(1)], &[v(2), v(0), v(2)],
            // H channel around C atom, gradient w.r.t. C positions
            &[v(2), v(1), v(0)], &[v(2), v(1), v(1)], &[v(2), v(1), v(2)],
            // C channel around C atom, gradient w.r.t. C positions
            &[v(3), v(1), v(0)], &[v(3), v(1), v(1)], &[v(3), v(1), v(2)],
        ]);
    }

    #[test]
    fn self_contribution_partial_gradients() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        samples.add(&[v(1), v(0), v(123456), v(1)]);
        samples.add(&[v(0), v(0), v(1), v(6)]);
        samples.add(&[v(1), v(1), v(1), v(1)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["CH", "water"]);
        let builder = TwoBodiesSpeciesSamples::with_self_contribution(2.0);

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.count(), 21);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around O in water, gradient w.r.t. O positions
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            // H channel around O in water, gradient w.r.t. H1 positions
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            // H channel around O in water, gradient w.r.t. H2 positions
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            // C channel around H in CH, gradient w.r.t. H positions
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            // C channel around H in CH, gradient w.r.t. C positions
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            // H channel around H1 in water, gradient w.r.t. H1 positions
            &[v(2), v(1), v(0)], &[v(2), v(1), v(1)], &[v(2), v(1), v(2)],
            // H channel around H1 in water, gradient w.r.t. H2 positions
            &[v(2), v(2), v(0)], &[v(2), v(2), v(1)], &[v(2), v(2), v(2)],
        ]);
    }
}
