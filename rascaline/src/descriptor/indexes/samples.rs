use indexmap::IndexSet;

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

    fn gradients_names(&self) -> Option<Vec<&str>> {
        let mut names = self.names();
        names.extend_from_slice(&["atom", "spatial"]);
        return Some(names);
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

        let mut gradients = IndexesBuilder::new(self.gradients_names().expect("gradient names"));
        for value in samples.iter() {
            let system = value[0];
            for atom in 0..systems[system.usize()].size()? {
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(0)]);
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(1)]);
                gradients.add(&[system, IndexValue::from(atom), IndexValue::from(2)]);
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

    fn gradients_names(&self) -> Option<Vec<&str>> {
        let mut names = self.names();
        names.extend_from_slice(&["neighbor", "spatial"]);
        return Some(names);
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

        // We need IndexSet to yield the indexes in the right order, i.e. the
        // order corresponding to whatever was passed in sample
        let mut indexes = IndexSet::new();
        for requested in samples {
            let i_system = requested[0];
            let center = requested[1].usize();
            let system = &mut *systems[i_system.usize()];
            system.compute_neighbors(self.cutoff)?;

            for pair in system.pairs_containing(center)? {
                if pair.first == center {
                    indexes.insert((i_system, pair.first, pair.second));
                } else if pair.second == center {
                    indexes.insert((i_system, pair.second, pair.first));
                }
            }
        }

        let mut gradients = IndexesBuilder::new(self.gradients_names().expect("gradients names"));
        for (structure, atom, neighbor) in indexes {
            let atom = IndexValue::from(atom);
            let neighbor = IndexValue::from(neighbor);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(0)]);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(1)]);
            gradients.add(&[structure, atom, neighbor, IndexValue::from(2)]);
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
        let builder = StructureSamples;
        assert_eq!(builder.names(), &["structure"]);

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples.names(), builder.names());
        assert_eq!(samples.count(), 3);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![&[v!(0)], &[v!(1)], &[v!(2)]]);
    }

    #[test]
    fn structure_gradient() {
        let mut systems = test_systems(&["methane", "water"]).boxed();

        let (_, gradients) = StructureSamples.with_gradients(&mut systems).unwrap();
        let gradients = gradients.unwrap();
        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), &["structure", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // methane
            &[v!(0), v!(0), v!(0)], &[v!(0), v!(0), v!(1)], &[v!(0), v!(0), v!(2)],
            &[v!(0), v!(1), v!(0)], &[v!(0), v!(1), v!(1)], &[v!(0), v!(1), v!(2)],
            &[v!(0), v!(2), v!(0)], &[v!(0), v!(2), v!(1)], &[v!(0), v!(2), v!(2)],
            &[v!(0), v!(3), v!(0)], &[v!(0), v!(3), v!(1)], &[v!(0), v!(3), v!(2)],
            &[v!(0), v!(4), v!(0)], &[v!(0), v!(4), v!(1)], &[v!(0), v!(4), v!(2)],
            // water
            &[v!(1), v!(0), v!(0)], &[v!(1), v!(0), v!(1)], &[v!(1), v!(0), v!(2)],
            &[v!(1), v!(1), v!(0)], &[v!(1), v!(1), v!(1)], &[v!(1), v!(1), v!(2)],
            &[v!(1), v!(2), v!(0)], &[v!(1), v!(2), v!(1)], &[v!(1), v!(2), v!(2)],
        ]);
    }

    #[test]
    fn partial_structure_gradient() {
        let mut samples = IndexesBuilder::new(vec!["structure"]);
        samples.add(&[v!(2)]);
        samples.add(&[v!(0)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["water", "methane", "water", "methane"]).boxed();
        let gradients = StructureSamples.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), &["structure", "atom", "spatial"]);
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // water #2
            &[v!(2), v!(0), v!(0)], &[v!(2), v!(0), v!(1)], &[v!(2), v!(0), v!(2)],
            &[v!(2), v!(1), v!(0)], &[v!(2), v!(1), v!(1)], &[v!(2), v!(1), v!(2)],
            &[v!(2), v!(2), v!(0)], &[v!(2), v!(2), v!(1)], &[v!(2), v!(2), v!(2)],
            // water #1
            &[v!(0), v!(0), v!(0)], &[v!(0), v!(0), v!(1)], &[v!(0), v!(0), v!(2)],
            &[v!(0), v!(1), v!(0)], &[v!(0), v!(1), v!(1)], &[v!(0), v!(1), v!(2)],
            &[v!(0), v!(2), v!(0)], &[v!(0), v!(2), v!(1)], &[v!(0), v!(2), v!(2)],
        ]);
    }

    #[test]
    fn atoms() {
        let mut systems = test_systems(&["methane", "water"]).boxed();
        let builder = AtomSamples { cutoff: 2.0 };
        assert_eq!(builder.names(), &["structure", "center"]);

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(builder.names(), samples.names());
        assert_eq!(samples.count(), 8);
        assert_eq!(samples.iter().collect::<Vec<_>>(), vec![
            &[v!(0), v!(0)], &[v!(0), v!(1)], &[v!(0), v!(2)], &[v!(0), v!(3)], &[v!(0), v!(4)],
            &[v!(1), v!(0)], &[v!(1), v!(1)], &[v!(1), v!(2)],
        ]);
    }

    #[test]
    fn atom_gradients() {
        let mut systems = test_systems(&["methane"]).boxed();
        let builder = AtomSamples { cutoff: 1.5 };
        assert_eq!(builder.gradients_names().unwrap(), &["structure", "center", "neighbor", "spatial"]);

        let (_, gradients) = builder.with_gradients(&mut systems).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.count(), 24);
        assert_eq!(gradients.names(), builder.gradients_names().unwrap());
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // Only C-H neighbors are within 1.3 A
            // C center
            &[v!(0), v!(0), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(2)],

            &[v!(0), v!(0), v!(2), v!(0)],
            &[v!(0), v!(0), v!(2), v!(1)],
            &[v!(0), v!(0), v!(2), v!(2)],

            &[v!(0), v!(0), v!(3), v!(0)],
            &[v!(0), v!(0), v!(3), v!(1)],
            &[v!(0), v!(0), v!(3), v!(2)],

            &[v!(0), v!(0), v!(4), v!(0)],
            &[v!(0), v!(0), v!(4), v!(1)],
            &[v!(0), v!(0), v!(4), v!(2)],
            // H centers
            &[v!(0), v!(1), v!(0), v!(0)],
            &[v!(0), v!(1), v!(0), v!(1)],
            &[v!(0), v!(1), v!(0), v!(2)],

            &[v!(0), v!(2), v!(0), v!(0)],
            &[v!(0), v!(2), v!(0), v!(1)],
            &[v!(0), v!(2), v!(0), v!(2)],

            &[v!(0), v!(3), v!(0), v!(0)],
            &[v!(0), v!(3), v!(0), v!(1)],
            &[v!(0), v!(3), v!(0), v!(2)],

            &[v!(0), v!(4), v!(0), v!(0)],
            &[v!(0), v!(4), v!(0), v!(1)],
            &[v!(0), v!(4), v!(0), v!(2)],
        ]);
    }

    #[test]
    fn partial_atom_gradient() {
        let mut samples = IndexesBuilder::new(vec!["structure", "center"]);
        // out of order values to ensure the gradients are also out of order
        samples.add(&[v!(0), v!(2)]);
        samples.add(&[v!(0), v!(0)]);
        let samples = samples.finish();

        let mut systems = test_systems(&["methane"]).boxed();
        let builder = AtomSamples { cutoff: 1.5 };
        assert_eq!(builder.gradients_names().unwrap(), &["structure", "center", "neighbor", "spatial"]);

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        let gradients = gradients.unwrap();

        assert_eq!(gradients.names(), builder.gradients_names().unwrap());
        assert_eq!(gradients.iter().collect::<Vec<_>>(), vec![
            // H centers
            &[v!(0), v!(2), v!(0), v!(0)],
            &[v!(0), v!(2), v!(0), v!(1)],
            &[v!(0), v!(2), v!(0), v!(2)],
            // C center
            &[v!(0), v!(0), v!(1), v!(0)],
            &[v!(0), v!(0), v!(1), v!(1)],
            &[v!(0), v!(0), v!(1), v!(2)],

            &[v!(0), v!(0), v!(2), v!(0)],
            &[v!(0), v!(0), v!(2), v!(1)],
            &[v!(0), v!(0), v!(2), v!(2)],

            &[v!(0), v!(0), v!(3), v!(0)],
            &[v!(0), v!(0), v!(3), v!(1)],
            &[v!(0), v!(0), v!(3), v!(2)],

            &[v!(0), v!(0), v!(4), v!(0)],
            &[v!(0), v!(0), v!(4), v!(1)],
            &[v!(0), v!(0), v!(4), v!(2)],
        ]);
    }
}
