use std::collections::BTreeSet;

use crate::{Error, System};
use super::super::{SamplesIndexes, Indexes, IndexesBuilder, IndexValue};

/// `StructureSpeciesSamples` is used to represents samples corresponding to
/// full structures, where each chemical species in the structure is represented
/// separately.
///
/// The base set of indexes contains `structure` and `species` the  gradient
/// indexes also contains the `atom` inside the structure with respect to which
/// the gradient is taken and the `spatial` (i.e. x/y/z) index.
pub struct StructureSpeciesSamples;

impl SamplesIndexes for StructureSpeciesSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "species"]
    }

    #[time_graph::instrument(name = "StructureSpeciesSamples::indexes")]
    fn indexes(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        let mut indexes = IndexesBuilder::new(self.names());
        for (i_system, system) in systems.iter().enumerate() {
            for &species in system.species()?.iter().collect::<BTreeSet<_>>() {
                indexes.add(&[
                    IndexValue::from(i_system), IndexValue::from(species)
                ]);
            }
        }
        return Ok(indexes.finish());
    }

    #[time_graph::instrument(name = "StructureSpeciesSamples::gradients_for")]
    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        let mut gradients = IndexesBuilder::new(vec!["structure", "species", "atom", "spatial"]);
        for value in samples.iter() {
            let i_system = value[0];
            let alpha = value[1];

            let system = &systems[i_system.usize()];
            let species = system.species()?;
            for (i_atom, &species) in species.iter().enumerate() {
                // only atoms with the same species participate to the gradient
                if species == alpha.usize() {
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(0)]);
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(1)]);
                    gradients.add(&[i_system, alpha, IndexValue::from(i_atom), IndexValue::from(2)]);
                }
            }
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
    fn structure() {
        let mut systems = test_systems(&["methane", "methane", "water"]).boxed();
        let indexes = StructureSpeciesSamples.indexes(&mut systems).unwrap();
        assert_eq!(indexes.count(), 6);
        assert_eq!(indexes.names(), &["structure", "species"]);
        assert_eq!(indexes.iter().collect::<Vec<_>>(), vec![
            &[v!(0), v!(1)], &[v!(0), v!(6)],
            &[v!(1), v!(1)], &[v!(1), v!(6)],
            &[v!(2), v!(1)], &[v!(2), v!(123456)],
        ]);
    }

    #[test]
    fn structure_gradient() {
        let mut systems = test_systems(&["CH", "water"]).boxed();
        let (_, gradients) = StructureSpeciesSamples.with_gradients(&mut systems).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 15);
        assert_eq!(gradients.names(), &["structure", "species", "atom", "spatial"]);

        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel in CH
            &[v!(0), v!(1), v!(0), v!(0)], &[v!(0), v!(1), v!(0), v!(1)], &[v!(0), v!(1), v!(0), v!(2)],
            // C channel in CH
            &[v!(0), v!(6), v!(1), v!(0)], &[v!(0), v!(6), v!(1), v!(1)], &[v!(0), v!(6), v!(1), v!(2)],
            // H channel in water
            &[v!(1), v!(1), v!(1), v!(0)], &[v!(1), v!(1), v!(1), v!(1)], &[v!(1), v!(1), v!(1), v!(2)],
            &[v!(1), v!(1), v!(2), v!(0)], &[v!(1), v!(1), v!(2), v!(1)], &[v!(1), v!(1), v!(2), v!(2)],
            // O channel in water
            &[v!(1), v!(123456), v!(0), v!(0)], &[v!(1), v!(123456), v!(0), v!(1)], &[v!(1), v!(123456), v!(0), v!(2)],
        ]);
    }

    #[test]
    fn partial_structure_gradient() {
        let mut indexes = IndexesBuilder::new(vec!["structure", "species"]);
        indexes.add(&[v!(2), v!(1)]);
        indexes.add(&[v!(0), v!(6)]);
        let indexes = indexes.finish();

        let mut systems = test_systems(&["CH", "water", "CH"]).boxed();
        let gradients = StructureSpeciesSamples.gradients_for(&mut systems, &indexes).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.names(), &["structure", "species", "atom", "spatial"]);

        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H channel in CH #2
            &[v!(2), v!(1), v!(0), v!(0)],
            &[v!(2), v!(1), v!(0), v!(1)],
            &[v!(2), v!(1), v!(0), v!(2)],
            // C channel in CH #1
            &[v!(0), v!(6), v!(1), v!(0)],
            &[v!(0), v!(6), v!(1), v!(1)],
            &[v!(0), v!(6), v!(1), v!(2)],
        ]);
    }
}
