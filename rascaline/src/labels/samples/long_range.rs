use std::sync::Arc;

use equistore::{Labels, LabelsBuilder};

use crate::{Error, System};
use super::{SamplesBuilder, SpeciesFilter};


/// TODO: docs
/// TODO: find a better name
pub struct LongRangePerAtom {
    /// Filter for the central atom species
    pub species_center: SpeciesFilter,
    /// Filter for the neighbor atom species
    pub species_neighbor: SpeciesFilter,
    /// Should the central atom be considered it's own neighbor?
    pub self_pairs: bool,
}

impl SamplesBuilder for LongRangePerAtom {
    fn samples_names() -> Vec<&'static str> {
        vec!["structure", "center"]
    }

    fn samples(&self, systems: &mut [Box<dyn System>]) -> Result<Arc<Labels>, Error> {
        assert!(self.self_pairs, "self.self_pairs = false is not implemented");

        let mut builder = LabelsBuilder::new(Self::samples_names());
        for (system_i, system) in systems.iter_mut().enumerate() {
            let species = system.species()?;

            match &self.species_neighbor {
                SpeciesFilter::Any => {
                    for (center_i, &species_center) in species.iter().enumerate() {
                        if self.species_center.matches(species_center) {
                            builder.add(&[system_i, center_i]);
                        }
                    }
                }
                // we want to take all centers of the right species iff
                // there is at least one atom in the system matching the
                // neighbor_species
                selection => {
                    let mut has_matching_neighbor = false;
                    for &species_neighbor in species {
                        if selection.matches(species_neighbor) {
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
                },
            }
        }

        return Ok(Arc::new(builder.finish()));
    }

    fn gradients_for(&self, systems: &mut [Box<dyn System>], samples: &Labels) -> Result<Arc<Labels>, Error> {
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

        return Ok(Arc::new(builder.finish()));
    }
}

#[cfg(test)]
mod tests {
    // TODO
}
