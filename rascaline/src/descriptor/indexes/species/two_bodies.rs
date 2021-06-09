use std::collections::BTreeSet;

use indexmap::IndexSet;

use crate::{Error, System};
use super::super::{SamplesIndexes, Indexes, IndexesBuilder, IndexValue};

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

impl SamplesIndexes for TwoBodiesSpeciesSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center", "species_center", "species_neighbor"]
    }

    #[time_graph::instrument(name = "TwoBodiesSpeciesSamples::indexes")]
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
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

    #[time_graph::instrument(name = "TwoBodiesSpeciesSamples::gradients_for")]
    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        // We need IndexSet to yield the indexes in the right order, i.e. the
        // order corresponding to whatever was passed in `samples`
        let mut indexes = IndexSet::new();
        for requested in samples {
            let i_system = requested[0];
            let center = requested[1].usize();
            let species_center = requested[2].usize();
            let species_neighbor = requested[3].usize();

            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff)?;

            if species_neighbor == species_center && self.self_contribution {
                indexes.insert((i_system, center, species_center, species_center, center));
            }

            let species = system.species()?;
            for pair in system.pairs_containing(center)? {
                let neighbor = if pair.first == center { pair.second } else { pair.first };

                if species[neighbor] != species_neighbor {
                    continue;
                }

                indexes.insert((i_system, center, species_center, species_neighbor, center));
                indexes.insert((i_system, center, species_center, species_neighbor, neighbor));
            }
        }

        let mut gradients = IndexesBuilder::new(vec![
            "structure", "center", "species_center", "species_neighbor",
            "neighbor", "spatial"
        ]);
        for (system, c, a, b, n) in indexes {
            let center = IndexValue::from(c);
            let alpha = IndexValue::from(a);
            let beta = IndexValue::from(b);
            let neighbor = IndexValue::from(n);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(0)]);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(1)]);
            gradients.add(&[system, center, alpha, beta, neighbor, IndexValue::from(2)]);
        }

        return Ok(Some(gradients.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_systems;

    /// Convenience macro to create IndexValue
    macro_rules! v {
        ($value: expr) => {
            crate::descriptor::indexes::IndexValue::from($value)
        };
    }

    #[test]
    fn samples() {
        let mut systems = test_systems(&["CH", "water"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::new(2.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 7);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // C channel around H in CH
            &[v!(0), v!(0), v!(1), v!(6)],
            // H channel around C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
            // H channel around O in water
            &[v!(1), v!(0), v!(123456), v!(1)],
            // H channel around the first H in water
            &[v!(1), v!(1), v!(1), v!(1)],
            // O channel around the first H in water
            &[v!(1), v!(1), v!(1), v!(123456)],
            // H channel around the second H in water
            &[v!(1), v!(2), v!(1), v!(1)],
            // O channel around the second H in water
            &[v!(1), v!(2), v!(1), v!(123456)],
        ]);
    }

    #[test]
    fn gradients() {
        let mut systems = test_systems(&["CH"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::new(2.0);
        let (_, gradients) = strategy.with_gradients(&mut systems).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 12);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // C channel around H, derivative w.r.t. H
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(2)],
            // C channel around H, derivative w.r.t. C
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // H channel around C, derivative w.r.t. C
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(0)],
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(1)],
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(2)],
            // H channel around C, derivative w.r.t. H
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(0)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(1)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(2)]
        ]);
    }

    #[test]
    fn partial_gradients() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        indexes.add(&[v!(1), v!(0), v!(123456), v!(1)]);
        indexes.add(&[v!(0), v!(0), v!(1), v!(6)]);
        indexes.add(&[v!(1), v!(1), v!(1), v!(1)]);
        let indexes = indexes.finish();

        let mut systems = test_systems(&["CH", "water"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::new(2.0);
        let gradients = strategy.gradients_for(&mut systems, &indexes).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 21);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around O in water, derivative w.r.t. O
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(2)],
            // H channel around O in water, derivative w.r.t. H1
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(2)],
            // H channel around O in water, derivative w.r.t. H2
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(2)],
            // C channel around H in CH, derivative w.r.t. H
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(2)],
            // C channel around H in CH, derivative w.r.t. C
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // H channel around H1 in water, derivative w.r.t. H1
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(2)],
            // H channel around H1 in water, derivative w.r.t. H2
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(2)]
        ]);
    }

    #[test]
    fn self_contribution() {
        let mut systems = test_systems(&["CH"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::new(2.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 2);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // C channel around H in CH
            &[v!(0), v!(0), v!(1), v!(6)],
            // H channel around C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
        ]);

        let strategy = TwoBodiesSpeciesSamples::with_self_contribution(2.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 4);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H channel around H in CH
            &[v!(0), v!(0), v!(1), v!(1)],
            // C channel around H in CH
            &[v!(0), v!(0), v!(1), v!(6)],
            // H channel around C in CH
            &[v!(0), v!(1), v!(6), v!(1)],
            // C channel around C in CH
            &[v!(0), v!(1), v!(6), v!(6)],
        ]);

        // we get entries even without proper neighbors
        let strategy = TwoBodiesSpeciesSamples::with_self_contribution(1.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 2);
        assert_eq!(indexes.names(), &["structure", "center", "species_center", "species_neighbor"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H channel around H in CH
            &[v!(0), v!(0), v!(1), v!(1)],
            // C channel around C in CH
            &[v!(0), v!(1), v!(6), v!(6)],
        ]);
    }

    #[test]
    fn self_contribution_gradients() {
        let mut systems = test_systems(&["CH"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::with_self_contribution(2.0);
        let (_, gradients) = strategy.with_gradients(&mut systems).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 18);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around H atom, gradient w.r.t. H positions
            &[v!(0), v!(0), v!(1), v!(1), v!(0), v!(0)],
            &[v!(0), v!(0), v!(1), v!(1), v!(0), v!(1)],
            &[v!(0), v!(0), v!(1), v!(1), v!(0), v!(2)],
            // C channel around H atom, gradient w.r.t. H positions
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(2)],
            // C channel around H atom, gradient w.r.t. C positions
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // H channel around C atom, gradient w.r.t. C positions
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(0)],
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(1)],
            &[v!(0), v!(1), v!(6), v!(1), v!(1), v!(2)],
            // H channel around C atom, gradient w.r.t. H positions
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(0)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(1)],
            &[v!(0), v!(1), v!(6), v!(1), v!(0), v!(2)],
            // C channel around C atom, gradient w.r.t. C positions
            &[v!(0), v!(1), v!(6), v!(6), v!(1), v!(0)],
            &[v!(0), v!(1), v!(6), v!(6), v!(1), v!(1)],
            &[v!(0), v!(1), v!(6), v!(6), v!(1), v!(2)],
        ]);
    }

    #[test]
    fn self_contribution_partial_gradients() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "center", "species_center", "species_neighbor"]);
        indexes.add(&[v!(1), v!(0), v!(123456), v!(1)]);
        indexes.add(&[v!(0), v!(0), v!(1), v!(6)]);
        indexes.add(&[v!(1), v!(1), v!(1), v!(1)]);
        let indexes = indexes.finish();

        let mut systems = test_systems(&["CH", "water"]).boxed();
        let strategy = TwoBodiesSpeciesSamples::with_self_contribution(2.0);
        let gradients = strategy.gradients_for(&mut systems, &indexes).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 21);
        assert_eq!(gradients.names(), &["structure", "center", "species_center", "species_neighbor", "neighbor", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel around O in water, gradient w.r.t. O positions
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(0), v!(2)],
            // H channel around O in water, gradient w.r.t. H1 positions
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(1), v!(2)],
            // H channel around O in water, gradient w.r.t. H2 positions
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(0)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(1)],
            &[v!(1), v!(0), v!(123456), v!(1), v!(2), v!(2)],
            // C channel around H in CH, gradient w.r.t. H positions
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(0), v!(2)],
            // C channel around H in CH, gradient w.r.t. C positions
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(6), v!(1), v!(2)],
            // H channel around H1 in water, gradient w.r.t. H1 positions
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(1), v!(2)],
            // H channel around H1 in water, gradient w.r.t. H2 positions
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(0)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(1)],
            &[v!(1), v!(1), v!(1), v!(1), v!(2), v!(2)],
        ]);
    }
}
