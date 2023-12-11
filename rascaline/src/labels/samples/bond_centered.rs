use std::collections::{BTreeSet,BTreeMap};

use metatensor::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, SpeciesFilter};
use crate::systems::BATripletNeighborList;


/// `SampleBuilder` for bond-centered representations. This will create one
/// sample for each pair of atoms (within a spherical cutoff to each other),
/// optionally filtering on the bond's atom species. The samples names are
/// (structure", "first_center", "second_center", "bond_i").
/// (with species(first_center)<=species(second_center))
///
/// Positions gradient samples include all atoms within a spherical cutoff to the bond center,
/// optionally filtering on the neighbor atom species.
pub struct BondCenteredSamples<'a> {
    /// spherical cutoff radius used to construct the atom-centered environments
    pub cutoffs: [f64;2],
    /// Filter for the central atom species
    pub species_center_1: SpeciesFilter,
    pub species_center_2: SpeciesFilter,
    /// Filter for the neighbor atom species
    pub species_neighbor: SpeciesFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_contributions: bool,
    pub raw_triplets: &'a BATripletNeighborList,
}

impl<'a> BondCenteredSamples<'a> {
    pub fn bond_cutoff(&self) -> f64 {
        self.cutoffs[0]
    }
    pub fn third_cutoff(&self) -> f64 {
        self.cutoffs[1]
    }
}

impl<'a> SamplesBuilder for BondCenteredSamples<'a> {
    fn sample_names() -> Vec<&'static str> {
        // bond_i is needed in case we have several bonds with the same atoms (periodic boundaries)
        vec!["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"]
    }

    fn samples(&self, systems: &mut [System]) -> Result<Labels, Error> {
        assert!(
            self.bond_cutoff() > 0.0 && self.bond_cutoff().is_finite() && self.third_cutoff() > 0.0 && self.third_cutoff().is_finite(),
            "cutoffs must be positive for BondCenteredSamples"
        );
        let mut builder = LabelsBuilder::new(Self::sample_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            self.raw_triplets.ensure_computed_for_system(system)?;
            let species = system.species()?;
            
            let mut center_cache: BTreeMap<(usize,usize,[i32;3]), BTreeSet<i32>> = BTreeMap::new();
            
            match (&self.species_center_1, &self.species_center_2) {
                (SpeciesFilter::Any, SpeciesFilter::Any) => {
                    for triplet in self.raw_triplets.get_for_system(system)? {
                        if self.self_contributions || (!triplet.is_self_contrib){
                            center_cache.entry((triplet.atom_i, triplet.atom_j, triplet.bond_cell_shift))
                                .or_insert_with(BTreeSet::new)
                                .insert(species[triplet.atom_k]);
                        }
                    }
                }
                (SpeciesFilter::AllOf(_),_)|(_,SpeciesFilter::AllOf(_)) =>
                    panic!("Cannot use Species::AllOf on BondCenteredSamples.center_species"),
                (SpeciesFilter::Single(s1), SpeciesFilter::Single(s2)) => {
                    let species_set = BTreeSet::from_iter(species.iter());
                    for s3 in species_set {
                        for triplet in self.raw_triplets.get_per_system_per_species(system, *s1, *s2, *s3)? {
                            if !self.self_contributions && triplet.is_self_contrib {
                                continue;
                            }
                            center_cache.entry((triplet.atom_i, triplet.atom_j, triplet.bond_cell_shift))
                                .or_insert_with(BTreeSet::new)
                                .insert(species[triplet.atom_k]);
                        }
                    }
                    
                },
                (selection_1, selection_2) => {
                    for (center_i, &species_center) in species.iter().enumerate() {
                        if !selection_1.matches(species_center) {
                            continue;
                        }
                        for (center_j, &species_center_2) in species.iter().enumerate() {
                            if !selection_2.matches(species_center_2) {
                                continue;
                            }
                            for triplet in self.raw_triplets.get_per_system_per_center(system, center_i, center_j)? {
                                if !self.self_contributions && triplet.is_self_contrib {
                                    continue;
                                }
                                center_cache.entry((triplet.atom_i, triplet.atom_j, triplet.bond_cell_shift))
                                .or_insert_with(BTreeSet::new)
                                .insert(species[triplet.atom_k]);
                            }
                        }
                    }
                }
            }
            match &self.species_neighbor {
                SpeciesFilter::Any => {
                    for (center_1,center_2,cell_shft) in center_cache.keys() {
                        builder.add(&[system_i as i32,*center_1 as i32,*center_2 as i32, cell_shft[0],cell_shft[1],cell_shft[2]]);
                    }
                },
                SpeciesFilter::AllOf(requirements) => {
                    for ((center_1,center_2,cell_shft), neigh_set) in center_cache.iter() {
                        if requirements.is_subset(neigh_set) {
                            builder.add(&[system_i as i32,*center_1 as i32,*center_2 as i32, cell_shft[0],cell_shft[1],cell_shft[2]]);
                        }
                    }
                },
                SpeciesFilter::Single(requirement) => {
                    for ((center_1,center_2,cell_shft), neigh_set) in center_cache.iter() {
                        if neigh_set.contains(requirement) {
                            builder.add(&[system_i as i32,*center_1 as i32,*center_2 as i32, cell_shft[0],cell_shft[1],cell_shft[2]]);
                        }
                    }
                },
                SpeciesFilter::OneOf(requirements) => {
                    let requirements: BTreeSet<i32> = BTreeSet::from_iter(requirements.iter().map(|x|*x));
                    for ((center_1,center_2,cell_shft), neigh_set) in center_cache.iter() {
                        if neigh_set.intersection(&requirements).count()>0 {
                            builder.add(&[system_i as i32,*center_1 as i32,*center_2 as i32, cell_shft[0],cell_shft[1],cell_shft[2]]);
                        }
                    }
                },
            }
        }

        return Ok(builder.finish());
    }

    fn gradients_for(&self, systems: &mut [System], samples: &Labels) -> Result<Labels, Error> {
        assert!(
            self.bond_cutoff() > 0.0 && self.bond_cutoff().is_finite() && self.third_cutoff() > 0.0 && self.third_cutoff().is_finite(),
            "cutoffs must be positive for BondCenteredSamples"
        );
        assert_eq!(samples.names(), ["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"]);
        let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);

        // we could try to find a better way to estimate this, but in the worst
        // case this would only over-allocate a bit
        let average_neighbors_per_atom = 10;
        builder.reserve(average_neighbors_per_atom * samples.count());

        for (sample_i, [structure_i, center_1, center_2, clsh_a,clsh_b,clsh_c]) in samples.iter_fixed_size().enumerate() {
            let structure_i = structure_i.usize();
            let center_1 = center_1.usize();
            let center_2 = center_2.usize();
            let cell_shift = [clsh_a.i32(),clsh_b.i32(),clsh_c.i32()];

            let system = &mut systems[structure_i];
            self.raw_triplets.ensure_computed_for_system(system)?;
            let species = system.species()?;

            let mut grad_contributors = BTreeSet::new();
            grad_contributors.insert(center_1);
            grad_contributors.insert(center_2);

            for triplet in self.raw_triplets.get_per_system_per_center(system, center_1, center_2)? {
                if triplet.bond_cell_shift != cell_shift {
                    continue;
                }
                match &self.species_neighbor{
                    SpeciesFilter::Any | SpeciesFilter::AllOf(_) => {
                        // in both of those cases, the sample already has been validated, and all known neighbors contribute
                        grad_contributors.insert(triplet.atom_k);
                    },
                    neighbor_filter => {
                        if neighbor_filter.matches(species[triplet.atom_k]) {
                            grad_contributors.insert(triplet.atom_k); 
                        }
                    },   
                }
            }

            for contrib in grad_contributors{
                builder.add(&[sample_i, structure_i, contrib]);
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
        let raw = BATripletNeighborList {
            cutoffs: [2.0,2.0],
        };
        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Any,
            species_center_2: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Any,
            self_contributions: true,
            raw_triplets: &raw,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"],
            &[[0, 1, 0, 0,0,0], [1, 0, 1, 0,0,0], [1, 0, 2, 0,0,0], [1, 1, 2, 0,0,0]],
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
                [3, 1, 0], [3, 1, 1], [3, 1, 2],
            ],
        ));
    }

    #[test]
    fn filter_species_center() {
        let mut systems = test_systems(&["CH", "water"]);
        let raw = BATripletNeighborList {
            cutoffs: [2.0,2.0],
        };
        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Single(6),
            species_center_2: SpeciesFilter::Single(1),
            species_neighbor: SpeciesFilter::Any,
            self_contributions: true,
            raw_triplets: &raw,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"],
            &[[0, 1, 0, 0,0,0]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH
                [0, 0, 0], [0, 0, 1],
            ]
        ));
        
        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Single(1),
            species_center_2: SpeciesFilter::Single(1),
            species_neighbor: SpeciesFilter::Any,
            self_contributions: true,
            raw_triplets: &raw,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"],
            &[[1, 1, 2, 0,0,0]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in H2O
                [0, 1, 0], [0, 1, 1], [0, 1, 2],
            ]
        ));
    }

    #[test]
    fn filter_species_neighbor() {
        let mut systems = test_systems(&["CH", "water"]);
        let raw = BATripletNeighborList {
            cutoffs: [2.0,2.0],
        };
        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Any,
            species_center_2: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(1),
            self_contributions: true,
            raw_triplets: &raw,
        };

        let samples = builder.samples(&mut systems).unwrap();
        assert_eq!(samples, Labels::new(
            ["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"],
            &[[0, 1, 0, 0,0,0], [1, 0, 1, 0,0,0], [1, 0, 2, 0,0,0], [1, 1, 2, 0,0,0]],
        ));

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH w.r.t H atom only
                [0, 0, 0], [0, 0, 1],
                // gradients of atoms in water w.r.t H atoms only
                [1, 1, 0], [1, 1, 1], [1, 1, 2],
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
                [3, 1, 1], [3, 1, 2],
            ]
        ));

        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Any,
            species_center_2: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::OneOf(vec![1, 6]),
            self_contributions: true,
            raw_triplets: &raw,
        };

        let gradient_samples = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradient_samples, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of atoms in CH w.r.t C and H atoms
                [0, 0, 0], [0, 0, 1],
                // gradients of atoms in water w.r.t H atoms only
                [1, 1, 0], [1, 1, 1], [1, 1, 2],
                [2, 1, 0], [2, 1, 1], [2, 1, 2],
                [3, 1, 1], [3, 1, 2],
            ]
        ));
    }

    #[test]
    fn partial_gradients() {
        let samples = Labels::new(["structure", "first_center", "second_center", "cell_shift_a","cell_shift_b","cell_shift_c"], &[
            [1, 1, 0, 0,0,0],
            [0, 0, 1, 0,0,0],
            [1, 1, 2, 0,0,0],
        ]);

        let mut systems = test_systems(&["CH", "water"]);
        
        let raw = BATripletNeighborList {
            cutoffs: [2.0,2.0],
        };
        
        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Any,
            species_center_2: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(-42),
            self_contributions: true,
            raw_triplets: &raw,
        };

        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(["sample", "structure", "atom"], &[
            [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1],
            [2, 1, 0], [2, 1, 1], [2, 1, 2],
        ]));

        let builder = BondCenteredSamples {
            cutoffs: [2.0,2.0],
            species_center_1: SpeciesFilter::Any,
            species_center_2: SpeciesFilter::Any,
            species_neighbor: SpeciesFilter::Single(1),
            self_contributions: true,
            raw_triplets: &raw,
        };
        let gradients = builder.gradients_for(&mut systems, &samples).unwrap();
        assert_eq!(gradients, Labels::new(
            ["sample", "structure", "atom"],
            &[
                // gradients of first sample, O-H1 in water
                [0, 1, 0], [0, 1, 1], [0, 1, 2],
                // gradients of second sample, C-H in CH
                [1, 0, 0], [1, 0, 1],
                // gradients of third sample, H1-H2 in water
                [2, 1, 1], [2, 1, 2],
            ]
        ));
    }
}