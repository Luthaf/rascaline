use std::collections::BTreeSet;

use metatensor::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, AtomicTypeFilter};


/// `SampleBuilder` for atom-centered representation. This will create one
/// sample for each atom, optionally filtering on the central atom type. The
/// sample names are ("system", "atom").
///
/// Positions gradient samples include all atoms within a spherical cutoff,
/// optionally filtering on the neighbor atom type.
pub struct AtomCenteredSamples {
    /// spherical cutoff radius used to construct the atom-centered environments
    pub cutoff: f64,
    /// Filter for the central atom type
    pub center_type: AtomicTypeFilter,
    /// Filter for the neighbor atom type
    pub neighbor_type: AtomicTypeFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_pairs: bool,
}

impl SamplesBuilder for AtomCenteredSamples {
    fn sample_names() -> Vec<&'static str> {
        vec!["system", "atom"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite(), "cutoff must be positive for AtomCenteredSamples");
        let mut builder = LabelsBuilder::new(Self::sample_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            system.compute_neighbors(self.cutoff)?;
            let types = system.types()?;

            match &self.neighbor_type {
                AtomicTypeFilter::Any => {
                    for (center_i, &center_type) in types.iter().enumerate() {
                        if self.center_type.matches(center_type) {
                            builder.add(&[system_i, center_i]);
                        }
                    }
                }
                AtomicTypeFilter::AllOf(requested_types) => {
                    let mut neighbor_types = BTreeSet::new();
                    for (atom_i, &center_type) in types.iter().enumerate() {
                        if self.center_type.matches(center_type) {
                            for pair in system.pairs_containing(atom_i)? {
                                let neighbor = if pair.first == atom_i {
                                    pair.second
                                } else {
                                    pair.first
                                };

                                neighbor_types.insert(types[neighbor]);
                            }

                            if self.self_pairs {
                                neighbor_types.insert(center_type);
                            }

                            if requested_types.is_subset(&neighbor_types) {
                                builder.add(&[system_i, atom_i]);
                            }
                            neighbor_types.clear();
                        }
                    }
                }
                selection => {
                    let mut matching_atoms = BTreeSet::new();
                    for (atom_i, &center_type) in types.iter().enumerate() {
                        if self.center_type.matches(center_type) {
                            if self.self_pairs && selection.matches(center_type) {
                                matching_atoms.insert(atom_i);
                            }

                            for pair in system.pairs_containing(atom_i)? {
                                let neighbor = if pair.first == atom_i {
                                    pair.second
                                } else {
                                    pair.first
                                };

                                if selection.matches(types[neighbor]) {
                                    matching_atoms.insert(atom_i);
                                }
                            }
                        }
                    }

                    for atom_i in matching_atoms {
                        builder.add(&[system_i, atom_i]);
                    }
                }
            }
        }

        return Ok(builder.finish());
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Labels) -> Result<Labels, Error> {
        assert!(self.cutoff > 0.0 && self.cutoff.is_finite(), "cutoff must be positive for AtomCenteredSamples");
        assert_eq!(samples.names(), ["system", "atom"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "system", "atom"]);

        // we could try to find a better way to estimate this, but in the worst
        // case this would only over-allocate a bit
        let average_neighbors_per_atom = 10;
        builder.reserve(average_neighbors_per_atom * samples.count());

        for (sample_i, [system_i, center_i]) in samples.iter_fixed_size().enumerate() {
            let system_i = system_i.usize();
            let atom_i = center_i.usize();

            let system = &mut systems[system_i];
            system.compute_neighbors(self.cutoff)?;
            let types = system.types()?;

            let mut neighbors = BTreeSet::new();
            // gradient with respect to the position of the central atom
            if self.self_pairs && self.neighbor_type.matches(types[atom_i]) {
                neighbors.insert(atom_i);
            }

            for pair in system.pairs_containing(atom_i)? {
                let neighbor_i = if pair.first == atom_i {
                    pair.second
                } else {
                    debug_assert_eq!(pair.second, atom_i);
                    pair.first
                };

                if self.neighbor_type.matches(types[neighbor_i]) {
                    neighbors.insert(neighbor_i);
                    // if the neighbor type matches, the center will also
                    // contribute to gradients
                    neighbors.insert(atom_i);
                }
            }

            for neighbor in neighbors {
                builder.add(&[sample_i, system_i, neighbor]);
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
            center_type: AtomicTypeFilter::Any,
            neighbor_type: AtomicTypeFilter::Any,
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["system", "atom"],
            &[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "system", "atom"],
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
    fn filter_center_type() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
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

    #[test]
    fn filter_neighbor_type() {
        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            center_type: AtomicTypeFilter::Any,
            neighbor_type: AtomicTypeFilter::Single(1),
            self_pairs: true,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["system", "atom"],
            &[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "system", "atom"],
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
            center_type: AtomicTypeFilter::Any,
            neighbor_type: AtomicTypeFilter::OneOf(vec![1, 6]),
            self_pairs: true,
        };

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "system", "atom"],
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
        let samples = Labels::new(["system", "atom"], &[
            [1, 0],
            [0, 0],
            [1, 1],
        ]);

        let mut systems = test_systems(&["CH", "water"]);
        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            center_type: AtomicTypeFilter::Any,
            neighbor_type: AtomicTypeFilter::Single(-42),
            self_pairs: true,
        };

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(
            ["sample", "system", "atom"],
            &[[0, 1, 0], [2, 1, 0], [2, 1, 1]]
        ));

        let builder = AtomCenteredSamples {
            cutoff: 2.0,
            center_type: AtomicTypeFilter::Any,
            neighbor_type: AtomicTypeFilter::Single(1),
            self_pairs: true,
        };
        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(
            ["sample", "system", "atom"],
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
