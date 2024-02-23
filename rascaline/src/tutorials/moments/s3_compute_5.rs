use metatensor::{Labels, TensorMap, LabelsBuilder};

use crate::{System, Error};
use crate::labels::{CenterSingleNeighborsTypesKeys, KeysBuilder};
use crate::labels::{AtomCenteredSamples, SamplesBuilder, AtomicTypeFilter};
use crate::calculators::CalculatorBase;

// these are here just to make the code below compile
const first_sample_position: Option<usize> = None;
const second_sample_position: Option<usize> = None;
const first_block_id: Option<usize> = None;
const second_block_id: Option<usize> = None;
const n_neighbors_first: f64 = 0.0;
const n_neighbors_second: f64 = 0.0;

#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
}

impl CalculatorBase for GeometricMoments {
    fn name(&self) -> String {
        todo!()
    }

    fn parameters(&self) -> String {
        todo!()
    }

    fn cutoffs(&self) -> &[f64] {
        todo!()
    }

    fn keys(&self, systems: &mut [Box<dyn System>]) -> Result<Labels, Error> {
        todo!()
    }

    fn sample_names(&self) -> Vec<&str> {
        todo!()
    }

    fn samples(&self, keys: &Labels, systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        todo!()
    }

    fn positions_gradient_samples(&self, keys: &Labels, samples: &[Labels], systems: &mut [Box<dyn System>]) -> Result<Vec<Labels>, Error> {
        todo!()
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        todo!()
    }

    fn property_names(&self) -> Vec<&str> {
        todo!()
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        todo!()
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut TensorMap) -> Result<(), Error> {
        // ...

        // add this line
        let do_positions_gradients = descriptor.block_by_id(0).gradient("positions").is_some();

        for (system_i, system) in systems.iter_mut().enumerate() {
            // ...
            for pair in system.pairs()? {
                // ...

                if do_positions_gradients {
                    let mut moment_gradients = Vec::new();
                    for k in 0..=self.max_moment {
                        moment_gradients.push([
                            pair.vector[0] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                            pair.vector[1] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                            pair.vector[2] * k as f64 * f64::powi(pair.distance, (k as i32) - 2),
                        ]);
                    }

                    if let Some(sample_position) = first_sample_position {
                        let block_id = first_block_id.expect("we have a sample in this block");
                        let mut block = descriptor.block_mut_by_id(block_id);

                        let mut gradient = block.gradient_mut("positions").expect("missing gradient storage");
                        let gradient = gradient.data_mut();
                        let array = gradient.values.to_array_mut();

                        let gradient_wrt_second = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.second.into()
                        ]);
                        let gradient_wrt_self = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.first.into()
                        ]);

                        for (property_i, [k]) in gradient.properties.iter_fixed_size().enumerate() {
                            if let Some(sample_i) = gradient_wrt_second {
                                let grad = moment_gradients[k.usize()];
                                // There is one extra dimension in the gradients
                                // array compared to the values, accounting for
                                // each of the Cartesian directions.
                                array[[sample_i, 0, property_i]] += grad[0] / n_neighbors_first;
                                array[[sample_i, 1, property_i]] += grad[1] / n_neighbors_first;
                                array[[sample_i, 2, property_i]] += grad[2] / n_neighbors_first;
                            }

                            if let Some(sample_i) = gradient_wrt_self {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] -= grad[0] / n_neighbors_first;
                                array[[sample_i, 1, property_i]] -= grad[1] / n_neighbors_first;
                                array[[sample_i, 2, property_i]] -= grad[2] / n_neighbors_first;
                            }
                        }
                    }

                    if let Some(sample_position) = second_sample_position {
                        let block_id = second_block_id.expect("we have a sample in this block");
                        let mut block = descriptor.block_mut_by_id(block_id);

                        let mut gradient = block.gradient_mut("positions").expect("missing gradient storage");
                        let gradient = gradient.data_mut();
                        let array = gradient.values.to_array_mut();

                        let gradient_wrt_first = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.first.into()
                        ]);
                        let gradient_wrt_self = gradient.samples.position(&[
                            sample_position.into(), system_i.into(), pair.second.into()
                        ]);

                        for (property_i, [k]) in gradient.properties.iter_fixed_size().enumerate() {
                            if let Some(sample_i) = gradient_wrt_first {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] -= grad[0] / n_neighbors_second;
                                array[[sample_i, 1, property_i]] -= grad[1] / n_neighbors_second;
                                array[[sample_i, 2, property_i]] -= grad[2] / n_neighbors_second;
                            }

                            if let Some(sample_i) = gradient_wrt_self {
                                let grad = moment_gradients[k.usize()];
                                array[[sample_i, 0, property_i]] += grad[0] / n_neighbors_second;
                                array[[sample_i, 1, property_i]] += grad[1] / n_neighbors_second;
                                array[[sample_i, 2, property_i]] += grad[2] / n_neighbors_second;
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
    // [compute]
}
