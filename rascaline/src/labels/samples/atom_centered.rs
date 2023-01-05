use std::collections::BTreeSet;

use equistore::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, SpeciesFilter};


/// `SampleBuilder` for atom-centered representation. This will create one
/// sample for each atom, optionally filtering on the central atom species. The
/// samples names are ("structure", "center").
///
/// Positions gradient samples include all atoms within a spherical cutoff,
/// optionally filtering on the neighbor atom species.
pub struct AtomCenteredSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    pub cutoff: f64,
    /// Filter for the central atom species
    pub species_center: SpeciesFilter,
    /// Filter for the neighbor atom species
    pub species_neighbor: SpeciesFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_pairs: bool,
}

impl SamplesBuilder for AtomCenteredSamples {
    fn samples_names() -> Vec<&'static str> {
        vec!["structure", "center"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite(), "cutoff must be positive for AtomCenteredSamples");
        let mut builder = LabelsBuilder::new(Self::samples_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            match &self.species_neighbor {
                SpeciesFilter::Any => {
                    for (center_i, &species_center) in species.iter().enumerate() {
                        if self.species_center.matches(species_center) {
                            builder.add(&[system_i, center_i]);
                        }
                    }
                }
                SpeciesFilter::AllOf(requested_species) => {
                    let mut neighbor_species = BTreeSet::new();
                    for (center_i, &species_center) in species.iter().enumerate() {
                        if self.species_center.matches(species_center) {
                            for pair in system.pairs_containing(center_i)? {
                                let neighbor = if pair.first == center_i {
                                    pair.second
                                } else {
                                    pair.first
                                };

                                neighbor_species.insert(species[neighbor]);
                            }

                            if self.self_pairs {
                                neighbor_species.insert(species_center);
                            }

                            if requested_species.is_subset(&neighbor_species) {
                                builder.add(&[system_i, center_i]);
                            }
                            neighbor_species.clear();
                        }
                    }
                }
                selection => {
                    let mut matching_centers = BTreeSet::new();
                    for (center_i, &species_center) in species.iter().enumerate() {
                        if self.species_center.matches(species_center) {
                            if self.self_pairs && selection.matches(species_center) {
                                matching_centers.insert(center_i);
                            }

                            for pair in system.pairs_containing(center_i)? {
                                let neighbor = if pair.first == center_i {
                                    pair.second
                                } else {
                                    pair.first
                                };

                                if selection.matches(species[neighbor]) {
                                    matching_centers.insert(center_i);
                                }
                            }
                        }
                    }

                    for center in matching_centers {
                        builder.add(&[system_i, center]);
                    }
                }
            }
        }

        return Ok(builder.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Labels) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite(), "cutoff must be positive for AtomCenteredSamples");
        assert_eq!(samples.names(), ["structure", "center"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);

        // we could try to find a better way to estimate this, but in the worst
        // case this would only over-allocate a bit
        let average_neighbors_per_atom = 10;
        builder.reserve(average_neighbors_per_atom * samples.count());

        for (sample_i, [structure_i, center_i]) in samples.iter_fixed_size().enumerate() {
            let structure_i = structure_i.usize();
            let center_i = center_i.usize();

            let system = &mut systems[structure_i];
            system.compute_neighbors(self.cutoff)?;
            let species = system.species()?;

            let mut neighbors = BTreeSet::new();
            // gradient with respect to the position of the central atom
            if self.self_pairs && self.species_neighbor.matches(species[center_i]) {
                neighbors.insert(center_i);
            }

            for pair in system.pairs_containing(center_i)? {
                let neighbor_i = if pair.first == center_i {
                    pair.second
                } else {
                    debug_assert_eq!(pair.second, center_i);
                    pair.first
                };

                if self.species_neighbor.matches(species[neighbor_i]) {
                    neighbors.insert(neighbor_i);
                    // if the neighbor species matches, the center will also
                    // contribute to gradients
                    neighbors.insert(center_i);
                }
            }

            for neighbor in neighbors {
                builder.add(&[sample_i, structure_i, neighbor]);
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
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            species_center: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Any,
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "center"],
            &[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH
                [0, 0, 0], [0, 0, 1],
                [1, 0, 0], [1, 0, 1],
                // gradients of atoms in water
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
                [3, 1, 0], [3, 1, 1], [3, 1, 2],
                [4, 1, 0], [4, 1, 1], [4, 1, 2],
            ],
        ));
    }

    #[test]
    fn filter_species_center() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
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

    #[test]
    fn filter_species_neighbor() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            species_center: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(1),
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "center"],
            &[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH w.r.t H atom only
                [0, 0, 0], [0, 0, 1],
                [1, 0, 1],
                // gradients of atoms in water w.r.t H atoms only
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
                [3, 1, 1], [3, 1, 2],
                [4, 1, 1], [4, 1, 2],
            ]
        ));

        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            species_center: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::OneOf(vec![1, 6]),
            self_pairs: true,
        };

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH w.r.t C and H atoms
                [0, 0, 0], [0, 0, 1],
                [1, 0, 0], [1, 0, 1],
                // gradients of atoms in water w.r.t H atoms only
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
                [3, 1, 1], [3, 1, 2],
                [4, 1, 1], [4, 1, 2],
            ]
        ));
    }

    #[test]
    fn partial_gradients() {
        let samples = Labels::new(["structure", "center"], &[
            [1, 0],
            [0, 0],
            [1, 1],
        ]);

        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            species_center: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(-42),
            self_pairs: true,
        };

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(
            ["sample", "structure", "atom"],
            &[[0, 1, 0], [2, 1, 0], [2, 1, 1]]
        ));

        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            species_center: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(1),
            self_pairs: true,
        };
        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of first sample, O in water
                [0, 1, 0], [0, 1, 1], [0, 1, 2],
                // gradients of second sample, C in CH
                [1, 0, 0], [1, 0, 1],
                // gradients of third sample, H1 in water
                [2, 1, 1], [2, 1, 2]
            ]
        ));
    }
}
