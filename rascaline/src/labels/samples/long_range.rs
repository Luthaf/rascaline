use metatensor::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, AtomicTypeFilter};


/// Samples builder for long-range (i.e. infinite cutoff), per atom, two bodies
/// representation.
///
/// With this builder, all atoms are considered neighbors of all other atoms.
pub struct LongRangeSamplesPerAtom {
    /// Filter for the central atom type
    pub center_type: AtomicTypeFilter,
    /// Filter for the neighbor atom type
    pub neighbor_type: AtomicTypeFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_pairs: bool,
}

impl SamplesBuilder for LongRangeSamplesPerAtom {
    fn sample_names() -> Vec<&'static str> {
        vec!["system", "atom"]
    }

    fn samples(&self, systems: &mut [System]) -> Result<Labels, Error> {
        assert!(self.self_pairs, "self.self_pairs = false is not implemented");

        let mut builder = LabelsBuilder::new(Self::sample_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            let types = system.types()?;

            // we want to take all atoms matching `center_type` iff
            // there is at least one atom in the system matching
            // `neighbor_type`
            let mut has_matching_neighbor = false;
            for &neighbor_type in types {
                if self.neighbor_type.matches(neighbor_type) {
                    has_matching_neighbor = true;
                    break;
                }
            }

            if has_matching_neighbor {
                for (center_i, &center_type) in types.iter().enumerate() {
                    if self.center_type.matches(center_type) {
                        builder.add(&[system_i, center_i]);
                    }
                }
            }
        }

        return Ok(builder.finish());
    }

    fn gradients_for(&self, systems: &mut [System], samples: &Labels) -> Result<Labels, Error> {
        assert_eq!(samples.names(), ["system", "atom"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "system", "atom"]);

        for (sample_i, [system_i, center_i]) in samples.iter_fixed_size().enumerate() {
            let system_i = system_i.usize();

            let system = &mut systems[system_i];
            for (neighbor_i, &neighbor_type) in system.types()?.iter().enumerate() {
                if self.neighbor_type.matches(neighbor_type) || neighbor_i == center_i.usize() {
                    builder.add(&[sample_i, system_i, neighbor_i]);
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
            center_type: AtomicTypeFilter::Single(1),
            neighbor_type: AtomicTypeFilter::Any,
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["system", "atom"],
            &[[0, 1], [1, 1], [1, 2]],
        ));


        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "system", "atom"],
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
