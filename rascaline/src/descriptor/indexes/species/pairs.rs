use crate::{Error, System};
use super::super::{SamplesIndexes, Indexes, IndexesBuilder, IndexValue};

/// `PairSpeciesSamples` is used to represents samples associated with pairs of
/// atoms, including chemical species information.
///
/// The base set of indexes contains `structure`, `first` (first atom in the
/// pair), `second` (second atom in the pair), `pair_id`, `species_first` and
/// `species_second`.
///
/// The `pair_id` is a unique identifier allowing to differentiate multiple
/// pairs between the same two atoms (which can occur with periodic boundary
/// conditions and a large cutoff). For each pair, the same pair but in the
/// reverse direction will a `pair_id` with same value but opposite sign (i.e.
/// if the `pair_id` for a pair `i->j` is 42, the `pair_id` for `j->i` is -42).
///
/// If self-contributions (atoms can be their own neighbor) are enabled, the
/// corresponding `pair_id` is 0.
pub struct PairSpeciesSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
    /// Are atoms considered to be their own neighbor?
    self_contribution: bool,
}

impl PairSpeciesSamples {
    /// Create a new `PairSpeciesSamples` with the given `cutoff`, excluding
    /// self contributions.
    pub fn new(cutoff: f64) -> PairSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for PairSpeciesSamples");
        PairSpeciesSamples {
            cutoff: cutoff,
            self_contribution: false,
        }
    }

    /// Create a new `PairSpeciesSamples` with the given `cutoff`, including
    /// self contributions.
    pub fn with_self_contribution(cutoff: f64) -> PairSpeciesSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for PairSpeciesSamples");
        PairSpeciesSamples {
            cutoff: cutoff,
            self_contribution: true,
        }
    }
}

impl SamplesIndexes for PairSpeciesSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "first", "second", "pair_id", "species_first", "species_second"]
    }

    #[time_graph::instrument(name = "PairSpeciesSamples::indexes")]
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        let mut indexes = IndexesBuilder::new(self.names());

        for (i_system, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            if self.self_contribution {
                for (center, &species) in species.iter().enumerate() {
                    indexes.add(&[
                        IndexValue::from(i_system),
                        IndexValue::from(center),
                        IndexValue::from(center),
                        IndexValue::from(0),
                        IndexValue::from(species),
                        IndexValue::from(species)
                    ]);
                }
            }

            for (id, pair) in system.pairs()?.iter().enumerate() {
                let pair_id = (id + 1) as isize;
                let species_first = species[pair.first];
                let species_second = species[pair.second];

                indexes.add(&[
                    IndexValue::from(i_system),
                    IndexValue::from(pair.first),
                    IndexValue::from(pair.second),
                    IndexValue::from(pair_id),
                    IndexValue::from(species_first),
                    IndexValue::from(species_second)
                ]);

                indexes.add(&[
                    IndexValue::from(i_system),
                    IndexValue::from(pair.second),
                    IndexValue::from(pair.first),
                    IndexValue::from(-pair_id),
                    IndexValue::from(species_second),
                    IndexValue::from(species_first)
                ]);
            };
        }

        return Ok(indexes.finish());
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
        let strategy = PairSpeciesSamples::new(2.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 8);
        assert_eq!(indexes.names(), &["structure", "first", "second", "pair_id", "species_first", "species_second"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // C-H pair in CH
            &[v!(0), v!(0), v!(1), v!(1), v!(1), v!(6)],
            &[v!(0), v!(1), v!(0), v!(-1), v!(6), v!(1)],
            // O-H1 pair in water
            &[v!(1), v!(0), v!(1), v!(1), v!(123456), v!(1)],
            &[v!(1), v!(1), v!(0), v!(-1), v!(1), v!(123456)],
            // O-H2 pair in water
            &[v!(1), v!(0), v!(2), v!(2), v!(123456), v!(1)],
            &[v!(1), v!(2), v!(0), v!(-2), v!(1), v!(123456)],
            // H1-H2 pair in water
            &[v!(1), v!(1), v!(2), v!(3), v!(1), v!(1)],
            &[v!(1), v!(2), v!(1), v!(-3), v!(1), v!(1)]
        ]);
    }

    // #[test]
    // fn gradients() {
    //     todo!()
    // }

    // #[test]
    // fn partial_gradients() {
    //     todo!()
    // }

    #[test]
    fn self_contribution() {
        let mut systems = test_systems(&["CH", "water"]).boxed();
        let strategy = PairSpeciesSamples::with_self_contribution(2.0);
        let indexes = strategy.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 13);
        assert_eq!(indexes.names(), &["structure", "first", "second", "pair_id", "species_first", "species_second"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            // H self-pair in CH
            &[v!(0), v!(0), v!(0), v!(0), v!(1), v!(1)],
            // C self-pair in CH
            &[v!(0), v!(1), v!(1), v!(0), v!(6), v!(6)],
            // C-H pair in CH
            &[v!(0), v!(0), v!(1), v!(1), v!(1), v!(6)],
            &[v!(0), v!(1), v!(0), v!(-1), v!(6), v!(1)],
            // O self-pair in water
            &[v!(1), v!(0), v!(0), v!(0), v!(123456), v!(123456)],
            // H1 self-pair in water
            &[v!(1), v!(1), v!(1), v!(0), v!(1), v!(1)],
            // H2 self-pair in water
            &[v!(1), v!(2), v!(2), v!(0), v!(1), v!(1)],
            // O-H1 pair in water
            &[v!(1), v!(0), v!(1), v!(1), v!(123456), v!(1)],
            &[v!(1), v!(1), v!(0), v!(-1), v!(1), v!(123456)],
            // O-H2 pair in water
            &[v!(1), v!(0), v!(2), v!(2), v!(123456), v!(1)],
            &[v!(1), v!(2), v!(0), v!(-2), v!(1), v!(123456)],
            // H1-H2 pair in water
            &[v!(1), v!(1), v!(2), v!(3), v!(1), v!(1)],
            &[v!(1), v!(2), v!(1), v!(-3), v!(1), v!(1)]
        ]);
    }

    // #[test]
    // fn self_contribution_gradients() {
    //     todo!()
    // }

    // #[test]
    // fn self_contribution_partial_gradients() {
    //     todo!()
    // }
}
