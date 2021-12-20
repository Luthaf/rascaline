use crate::{System, Descriptor, Error};
use crate::descriptor::{SamplesBuilder, TwoBodiesSpeciesSamples};
use crate::descriptor::{Indexes, IndexesBuilder, IndexValue};
use crate::calculators::CalculatorBase;

#[derive(Clone, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
struct GeometricMoments {
    cutoff: f64,
    max_moment: usize,
    gradients: bool,
}

// declared as const to make the code compile
const species: &[usize] = &[];
const n_neighbors_first: f64 = 0.0;
const n_neighbors_second: f64 = 0.0;
const first_sample_position: Option<usize> = None;
const second_sample_position: Option<usize> = None;

impl CalculatorBase for GeometricMoments {
    fn name(&self) -> String {
        todo!()
    }

    fn get_parameters(&self) -> String {
        todo!()
    }

    fn compute_gradients(&self) -> bool {
        todo!()
    }

    fn features_names(&self) -> Vec<&str> {
        todo!()
    }

    fn features(&self) -> Indexes {
        todo!()
    }

    fn check_features(&self, indexes: &Indexes) -> Result<(), Error> {
        todo!()
    }

    fn samples_builder(&self) -> Box<dyn SamplesBuilder> {
        todo!()
    }

    // [compute]
    fn compute(&mut self, systems: &mut [Box<dyn System>], descriptor: &mut Descriptor) -> Result<(), Error> {
        // ...

        for (i_system, system) in systems.iter_mut().enumerate() {
            // ...

            for pair in system.pairs()? {
                // ...

                if self.gradients {
                    let gradients_samples = descriptor.gradients_samples.as_ref().expect("missing gradient samples");
                    let gradients = descriptor.gradients.as_mut().expect("missing gradient storage");

                    let mut first_gradient_position = None;
                    let mut first_gradient_self_position = None;
                    if let Some(i_first_sample) = first_sample_position {
                        // gradient of the descriptor around `pair.first` w.r.t. `pair.second`
                        first_gradient_position = gradients_samples.position(&[
                            IndexValue::from(i_first_sample),
                            IndexValue::from(pair.second),
                            IndexValue::from(0),
                        ]);

                        // gradient of the descriptor around `pair.first` w.r.t. `pair.first`
                        first_gradient_self_position = gradients_samples.position(&[
                            IndexValue::from(i_first_sample),
                            IndexValue::from(pair.first),
                            IndexValue::from(0),
                        ]);
                    }

                    let mut second_gradient_position = None;
                    let mut second_gradient_self_position = None;
                    if let Some(i_second_sample) = second_sample_position {
                        // gradient of the descriptor around `pair.second` w.r.t. `pair.first`
                        second_gradient_position = gradients_samples.position(&[
                            IndexValue::from(i_second_sample),
                            IndexValue::from(pair.first),
                            IndexValue::from(0),
                        ]);

                        // gradient of the descriptor around `pair.second` w.r.t. `pair.second`
                        second_gradient_self_position = gradients_samples.position(&[
                            IndexValue::from(i_second_sample),
                            IndexValue::from(pair.second),
                            IndexValue::from(0),
                        ]);
                    }

                    for (i_feature, feature) in descriptor.features.iter().enumerate() {
                        let k = feature[0].usize();
                        let grad_factor = k as f64 * f64::powi(pair.distance, (k as i32) - 2);

                        let grad_x = pair.vector[0] * grad_factor;
                        let grad_y = pair.vector[1] * grad_factor;
                        let grad_z = pair.vector[2] * grad_factor;

                        if let Some(i_first) = first_gradient_position {
                            gradients[[i_first + 0, i_feature]] += grad_x / n_neighbors_first;
                            gradients[[i_first + 1, i_feature]] += grad_y / n_neighbors_first;
                            gradients[[i_first + 2, i_feature]] += grad_z / n_neighbors_first;
                        }

                        if let Some(i_first_self) = first_gradient_self_position {
                            gradients[[i_first_self + 0, i_feature]] += -grad_x / n_neighbors_first;
                            gradients[[i_first_self + 1, i_feature]] += -grad_y / n_neighbors_first;
                            gradients[[i_first_self + 2, i_feature]] += -grad_z / n_neighbors_first;
                        }

                        if let Some(i_second) = second_gradient_position {
                            gradients[[i_second + 0, i_feature]] += -grad_x / n_neighbors_second;
                            gradients[[i_second + 1, i_feature]] += -grad_y / n_neighbors_second;
                            gradients[[i_second + 2, i_feature]] += -grad_z / n_neighbors_second;
                        }

                        if let Some(i_second_self) = second_gradient_self_position {
                            gradients[[i_second_self + 0, i_feature]] += grad_x / n_neighbors_second;
                            gradients[[i_second_self + 1, i_feature]] += grad_y / n_neighbors_second;
                            gradients[[i_second_self + 2, i_feature]] += grad_z / n_neighbors_second;
                        }
                    }
                }
            }
        }

        return Ok(());
    }
    // [compute]
}
