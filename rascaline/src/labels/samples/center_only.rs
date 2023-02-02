use std::sync::Arc;

use equistore::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, SpeciesFilter};


/// Samples builder each atom.
pub struct SamplesPerAtom {
    /// Filter for the atom species
    pub species_center: SpeciesFilter,
}

impl SamplesBuilder for SamplesPerAtom {
    fn samples_names() -> Vec<&'static str> {
        vec!["structure", "center"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Arc<Labels>, Error> {
        let mut builder = LabelsBuilder::new(Self::samples_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            let species = system.species()?;

            for (center_i, &species_center) in species.iter().enumerate() {
                if self.species_center.matches(species_center) {
                    builder.add(&[system_i, center_i]);
                }
            }
        }

        return Ok(Arc::new(builder.finish()));
    }

    fn gradients_for(&self, _: &mut [Box<dyn System>], samples: &Labels) -> Result<Arc<Labels>, Error> {
        assert_eq!(samples.names(), ["structure", "center"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);

        for (sample_i, [structure_i, center_i]) in samples.iter_fixed_size().enumerate() {
            builder.add(&[sample_i, structure_i.usize(), center_i.usize()]);
        }

        return Ok(Arc::new(builder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;

    #[test]
    fn all_samples() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = SamplesPerAtom {
            species_center: SpeciesFilter::Single(1),
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(*samples, Labels::new(
            ["structure", "center"],
            &[[0, 1], [1, 1], [1, 2]],
        ));


        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(*gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH
                [0, 0, 1],
                // gradients of O atoms in water
                [1, 1, 1], [2, 1, 2],
            ]
        ));
    }
}
