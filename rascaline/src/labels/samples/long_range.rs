use equistore::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, SpeciesFilter};


/// Samples builder for long-range (i.e. infinite cutoff), per atom, two bodies
/// representation.
///
/// With this builder, all atoms are considered neighbors of all other atoms.
pub struct LongRangeSamplesPerAtom {
    /// Filter for the central atom species
    pub species_center: SpeciesFilter,
    /// Filter for the neighbor atom species
    pub species_neighbor: SpeciesFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_pairs: bool,
}

impl SamplesBuilder for LongRangeSamplesPerAtom {
    fn samples_names() -> Vec<&'static str> {
        vec!["structure", "center"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.self_pairs, "self.self_pairs = false is not implemented");

        let mut builder = LabelsBuilder::new(Self::samples_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            let species = system.species()?;

            // we want to take all centers matching `species_center` iff
            // there is at least one atom in the system matching
            // `species_neighbor`
            let mut has_matching_neighbor = false;
            for &species_neighbor in species {
                if self.species_neighbor.matches(species_neighbor) {
                    has_matching_neighbor = true;
                    break;
                }
            }

            if has_matching_neighbor {
                for (center_i, &species_center) in species.iter().enumerate() {
                    if self.species_center.matches(species_center) {
                        builder.add(&[system_i, center_i]);
                    }
                }
            }
        }

        return Ok(builder.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Labels) -> Result<Labels, Error> {
        assert_eq!(samples.names(), ["structure", "center"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);

        for (sample_i, [structure_i, center_i]) in samples.iter_fixed_size().enumerate() {
            let structure_i = structure_i.usize();

            let system = &mut systems[structure_i];
            for (neighbor_i, &species_neighbor) in system.species()?.iter().enumerate() {
                if self.species_neighbor.matches(species_neighbor) || neighbor_i == center_i.usize() {
                    builder.add(&[sample_i, structure_i, neighbor_i]);
                }
            }
        }

        return Ok(builder.finish());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::test_utils::test_systems;

    #[test]
    fn all_samples() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = LongRangeSamplesPerAtom {
            species_center: SpeciesFilter::Single(1),
            species_neighbor: SpeciesFilter::Any,
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "center"],
            &[[0, 1], [1, 1], [1, 2]],
        ));


        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH
                [0, 0, 0], [0, 0, 1],
                // gradients of atoms in water
                [1, 1, 0], [1, 1, 1], [1, 1, 2],
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
            ]
        ));
    }
}
