use std::collections::BTreeSet;

use crate::{Error, System};
use super::{Indexes, IndexesBuilder, SamplesBuilder, IndexValue};

/// `StructureSamples` is used to represents samples corresponding to full
/// structures, each structure being described by a single features vector.
///
/// It does not contain any chemical species information, for this you should
/// use [`super::StructureSpeciesSamples`].
///
/// The base set of indexes contains only the `structure` index; the  gradient
/// indexes also contains the `atom` inside the structure with respect to which
/// the gradient is taken and the `spatial` (i.e. x/y/z) index.
pub struct StructureSamples;

impl SamplesBuilder for StructureSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        let mut indexes = IndexesBuilder::new(self.names());
        for system in 0..systems.len() {
            indexes.add(&[IndexValue::from(system)]);
        }
        return Ok(indexes.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        let mut gradients = IndexesBuilder::new(vec!["sample", "atom", "spatial"]);
        for (i_sample, sample) in samples.iter().enumerate() {
            let system = sample[0].usize();
            for atom in 0..systems[system].size()? {
                gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(0)]);
                gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(1)]);
                gradients.add(&[IndexValue::from(i_sample), IndexValue::from(atom), IndexValue::from(2)]);
            }
        }

        Ok(Some(gradients.finish()))
    }
}

/// `AtomSamples` is used to represents atom-centered environments, where
/// each atom in a structure is described with a feature vector based on other
/// atoms inside a sphere centered on the central atom.
///
/// This type of indexes does not contain any chemical species information, for
/// this you should use [`super::TwoBodiesSpeciesSamples`].
///
/// The base set of indexes contains `structure` and `center` (i.e. central atom
/// index inside the structure); the gradient indexes also contains the
/// `neighbor` inside the spherical cutoff with respect to which the gradient is
/// taken and the `spatial` (i.e x/y/z) index.
pub struct AtomSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    cutoff: f64,
}

impl AtomSamples {
    /// Create a new `AtomSamples` with the given cutoff.
    pub fn new(cutoff: f64) -> AtomSamples {
        assert!(cutoff > 0.0 && cutoff.is_finite(), "cutoff must be positive for AtomSamples");
        AtomSamples { cutoff }
    }
}

impl SamplesBuilder for AtomSamples {
    fn names(&self) -> Vec<&str> {
        vec!["structure", "center"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Indexes, Error> {
        let mut indexes = IndexesBuilder::new(self.names());
        for (i_system, system) in systems.iter().enumerate() {
            for center in 0..system.size()? {
                indexes.add(&[IndexValue::from(i_system), IndexValue::from(center)]);
            }
        }
        return Ok(indexes.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Indexes) -> Result<Option<Indexes>, Error> {
        assert_eq!(samples.names(), self.names());

        let mut indexes = BTreeSet::new();
        for (i_sample, sample) in samples.iter().enumerate() {
            let system_i = sample[0].usize();
            let center = sample[1].usize();

            let system = &mut *systems[system_i];
            system.compute_neighbors(self.cutoff)?;

            for pair in system.pairs_containing(center)? {
                if pair.first == center {
                    indexes.insert((i_sample, pair.second));
                } else if pair.second == center {
                    indexes.insert((i_sample, pair.first));
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;

    // small helper function to create IndexValue
    fn v(i: i32) -> IndexValue { IndexValue::from(i) }

    #[test]
    fn structure() {
        let mut systems = test_systems(&["methane", "water"]);
        let builder = StructureSamples;
        assert_eq!(builder.names(), &["structure"]);

        let (samples, gradients) = StructureSamples.with_gradients(&mut systems).unwrap();
        assert_eq!(samples.names(), builder.names());
        assert_eq!(samples.count(), 2);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![&[v(0)], &[v(1)]]);

        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // methane
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            &[v(0), v(3), v(0)], &[v(0), v(3), v(1)], &[v(0), v(3), v(2)],
            &[v(0), v(4), v(0)], &[v(0), v(4), v(1)], &[v(0), v(4), v(2)],
            // water
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            &[v(1), v(2), v(0)], &[v(1), v(2), v(1)], &[v(1), v(2), v(2)],
        ]);
    }

    #[test]
    fn partial_structure_gradient() {
        let mut samples = IndexesBuilder::new(vec!["structure"]);
        samples.add(&[v(2)]);
        samples.add(&[v(1)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["water", "methane", "water", "methane"]);
        let gradients = StructureSamples.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // water #2
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            // methane #1
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            &[v(1), v(2), v(0)], &[v(1), v(2), v(1)], &[v(1), v(2), v(2)],
            &[v(1), v(3), v(0)], &[v(1), v(3), v(1)], &[v(1), v(3), v(2)],
            &[v(1), v(4), v(0)], &[v(1), v(4), v(1)], &[v(1), v(4), v(2)],
        ]);
    }

    #[test]
    fn atoms() {
        let mut systems = test_systems(&["methane", "water"]);
        let builder = AtomSamples { cutoff: 1.5 };
        assert_eq!(builder.names(), &["structure", "center"]);

        let (samples, gradients) = builder.with_gradients(&mut systems).unwrap();
        assert_eq!(builder.names(), samples.names());
        assert_eq!(samples.count(), 8);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            // 5 atoms in methane
            &[v(0), v(0)],
            &[v(0), v(1)],
            &[v(0), v(2)],
            &[v(0), v(3)],
            &[v(0), v(4)],
            // 3 atoms in water
            &[v(1), v(0)],
            &[v(1), v(1)],
            &[v(1), v(2)],
        ]);

        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 36);
        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // Methane: only C-H neighbors are within 1.5 A
            // C center
            &[v(0), v(1), v(0)], &[v(0), v(1), v(1)], &[v(0), v(1), v(2)],
            &[v(0), v(2), v(0)], &[v(0), v(2), v(1)], &[v(0), v(2), v(2)],
            &[v(0), v(3), v(0)], &[v(0), v(3), v(1)], &[v(0), v(3), v(2)],
            &[v(0), v(4), v(0)], &[v(0), v(4), v(1)], &[v(0), v(4), v(2)],
            // H centers
            &[v(1), v(0), v(0)], &[v(1), v(0), v(1)], &[v(1), v(0), v(2)],
            &[v(2), v(0), v(0)], &[v(2), v(0), v(1)], &[v(2), v(0), v(2)],
            &[v(3), v(0), v(0)], &[v(3), v(0), v(1)], &[v(3), v(0), v(2)],
            &[v(4), v(0), v(0)], &[v(4), v(0), v(1)], &[v(4), v(0), v(2)],
            // Water: gradient around H1
            &[v(5), v(1), v(0)], &[v(5), v(1), v(1)], &[v(5), v(1), v(2)],
            // Water: gradient around H2
            &[v(5), v(2), v(0)], &[v(5), v(2), v(1)], &[v(5), v(2), v(2)],
            // Water: gradient around O
            &[v(6), v(0), v(0)], &[v(6), v(0), v(1)], &[v(6), v(0), v(2)],
            &[v(7), v(0), v(0)], &[v(7), v(0), v(1)], &[v(7), v(0), v(2)]
        ]);
    }

    #[test]
    fn partial_atom_gradient() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center"]);
        // out of order values to ensure the gradients are also out of order
        samples.add(&[v(0), v(2)]);
        samples.add(&[v(0), v(0)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["methane"]);
        let builder = AtomSamples { cutoff: 1.5 };

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["sample", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // Gradient around atom 2 (sample 0)
            &[v(0), v(0), v(0)], &[v(0), v(0), v(1)], &[v(0), v(0), v(2)],
            // Gradient around atom 0 (sample 1)
            &[v(1), v(1), v(0)], &[v(1), v(1), v(1)], &[v(1), v(1), v(2)],
            &[v(1), v(2), v(0)], &[v(1), v(2), v(1)], &[v(1), v(2), v(2)],
            &[v(1), v(3), v(0)], &[v(1), v(3), v(1)], &[v(1), v(3), v(2)],
            &[v(1), v(4), v(0)], &[v(1), v(4), v(1)], &[v(1), v(4), v(2)],
        ]);
    }
}
