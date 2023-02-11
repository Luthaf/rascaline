use std::sync::Arc;

use equistore::{Labels, LabelsBuilder};

use super::{SamplesBuilder, SpeciesFilter};
use crate::{Error, System};

/// Samples builder for structures only
pub struct Structures {
    /// Filter for the atom species. Only applies to gradients
    pub species_center: SpeciesFilter,
}

impl SamplesBuilder for Structures {
    fn samples_names() -> Vec<&'static str> {
        vec!["structure"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Arc<Labels>, Error> {
        let mut builder = LabelsBuilder::new(Self::samples_names());
        for system_i in 0..systems.len() {
            builder.add(&[system_i]);
        }

        return Ok(Arc::new(builder.finish()));
    }

    fn gradients_for(
        &self,
        systems: &mut [Box<dyn System>],
        samples: &Labels,
    ) -> Result<Arc<Labels>, Error> {
        assert_eq!(samples.names(), ["structure"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);

        for (system_i, system) in systems.iter_mut().enumerate() {
            let species = system.species()?;

            for (center_i, &species_center) in species.iter().enumerate() {
                if self.species_center.matches(species_center) {
                    builder.add(&[system_i, system_i, center_i]);
                }
            }
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
        let builder = Structures {
            species_center: SpeciesFilter::Single(1),
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(*samples, Labels::new(["structure"], &[[0], [1]],));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(
            *gradient_samples,
            Labels::new(
                ["sample", "structure", "atom"],
                &[
                    // gradients of atoms in CH
                    [0, 0, 1],
                    // gradients of O atoms in water
                    [1, 1, 1],
                    [1, 1, 2],
                ]
            )
        );
    }
}
