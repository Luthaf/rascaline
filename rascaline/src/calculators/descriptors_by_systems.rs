use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use ndarray::{ArrayViewMutD};
use once_cell::sync::Lazy;

use equistore::{TensorMap, TensorBlock, eqs_array_t};
use equistore::{LabelsBuilder, LabelValue};


/// Implementation of `equistore::Array` storing a view inside another array
///
/// This is relatively unsafe, and only viable for use inside this module.
struct UnsafeArrayViewMut {
    /// Shape of the sub-array
    shape: Vec<usize>,
    /// Pointer to the first element of the data. This point inside another
    /// array that is ASSUMED to stay alive for as long as this one does.
    ///
    /// We can not use lifetimes to track this assumption, since equistore
    /// requires `'static` lifetimes
    data: *mut f64,
}

static UNSAFE_ARRAY_VIEW_DATA_ORIGIN: Lazy<equistore::DataOrigin> = Lazy::new(|| {
    equistore::register_data_origin("rascaline.unsafe-array-view".into())
});

// SAFETY: `UnsafeArrayViewMut` can be transferred from one thread to another
unsafe impl Send for UnsafeArrayViewMut {}
// SAFETY: `UnsafeArrayViewMut` is Sync since there is no interior mutability
// (each array being a separate mutable view in the initial array)
unsafe impl Sync for UnsafeArrayViewMut {}

impl equistore::Array for UnsafeArrayViewMut {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn origin(&self) -> equistore::DataOrigin {
        *UNSAFE_ARRAY_VIEW_DATA_ORIGIN
    }

    fn create(&self, _: &[usize]) -> Box<dyn equistore::Array> {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }

    fn copy(&self) -> Box<dyn equistore::Array> {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }

    fn data(&self) -> &[f64] {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, _: &[usize]) {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }

    fn swap_axes(&mut self, _: usize, _: usize) {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }

    fn move_samples_from(
        &mut self,
        _: &dyn equistore::Array,
        _: &[equistore::eqs_sample_mapping_t],
        _: std::ops::Range<usize>,
    ) {
        unimplemented!("invalid operation on UnsafeArrayViewMut");
    }
}

/// Extract an array stored in the `TensorBloc` returned by `split_tensor_map_by_system`
pub fn array_mut_for_system(array: &mut eqs_array_t) -> ArrayViewMutD<'_, f64> {
    assert_eq!(
        array.origin().unwrap_or(equistore::eqs_data_origin_t(0)), *UNSAFE_ARRAY_VIEW_DATA_ORIGIN,
        "invalid array type"
    );

    let array = array.ptr.cast::<Box<dyn equistore::Array>>();
    let array: &mut UnsafeArrayViewMut = unsafe {
        (*array).as_any_mut().downcast_mut().expect("invalid array type")
    };

    // SAFETY: we checked that the slices do not overlap when creating
    // `UnsafeArrayViewMut` in split_by_system
    let slice = unsafe {
        std::slice::from_raw_parts_mut(array.data, array.shape.iter().product())
    };

    return ArrayViewMutD::from_shape(array.shape.clone(), slice).expect("wrong shape");
}

/// View inside a `TensorMap` corresponding to one system
pub struct TensorMapView<'a> {
    // all arrays in this TensorMap are `UnsafeArrayViewMut` with the lifetime
    // tracked by the marker
    data: TensorMap,
    marker: std::marker::PhantomData<&'a mut TensorMap>,
}

impl<'a> std::ops::Deref for TensorMapView<'a> {
    type Target = TensorMap;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> std::ops::DerefMut for TensorMapView<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}


/// Split a descriptor into multiple descriptors, one by system. The resulting
/// descriptors contain views inside the descriptor
#[allow(clippy::too_many_lines)]
pub fn split_tensor_map_by_system(descriptor: &mut TensorMap, n_systems: usize) -> Vec<TensorMapView<'_>> {
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct GradientPosition {
        positions: usize,
        cell: usize,
    }

    let mut descriptor_by_system = Vec::new();

    let mut values_end = vec![0; descriptor.keys().count()];
    let mut gradients_end = vec![GradientPosition { positions: 0, cell: 0 }; descriptor.keys().count()];
    for system_i in 0..n_systems {
        let blocks = descriptor.par_iter_mut()
            .zip_eq(&mut values_end)
            .zip_eq(&mut gradients_end)
            .map(|(((_, mut block), system_end), system_end_grad)| {
                let values_samples = &block.values().samples;
                let mut samples = LabelsBuilder::new(values_samples.names());
                let mut samples_mapping = BTreeMap::new();
                let mut structure_per_sample = vec![LabelValue::new(-1); values_samples.count()];

                let system_start = *system_end;
                for (sample_i, &[structure, center]) in values_samples.iter_fixed_size().enumerate().skip(system_start) {
                    structure_per_sample[sample_i] = structure;

                    if structure.usize() == system_i {
                        // this sample is part of to the current system
                        samples.add(&[structure, center]);
                        let new_sample = samples_mapping.len();
                        samples_mapping.insert(sample_i, new_sample);

                        *system_end += 1;
                    } else if structure.usize() > system_i {
                        // found the next system
                        break;
                    } else {
                        // structure.usize() < system_i
                        panic!("expected samples to be ordered by system, they are not");
                    }
                }

                let mut shape = Vec::new();

                let samples = Arc::new(samples.finish());
                shape.push(samples.count());

                let mut components = Vec::new();
                for component in &block.values().components {
                    components.push(Arc::clone(component));
                    shape.push(component.count());
                }

                let properties = Arc::clone(&block.values().properties);
                let n_properties = properties.count();
                shape.push(n_properties);

                let per_sample_size: usize = shape.iter().skip(1).product();
                let data_ptr = unsafe {
                    // SAFETY: this creates non-overlapping regions (from
                    // `data_ptr` to `data_ptr + shape.product()`.
                    //
                    // `per_sample_size * system_start` skips all the data
                    // associated with the previous systems.
                    block.values_mut().data.as_array_mut().as_mut_ptr().add(per_sample_size * system_start)
                };

                let data = UnsafeArrayViewMut {
                    shape: shape,
                    data: data_ptr,
                };
                let mut new_block = TensorBlock::new(
                    data, samples, components, properties
                ).expect("invalid TensorBlock");

                for (parameter, gradient) in block.gradients_mut() {
                    let system_end_grad = match &**parameter {
                        "positions" => &mut system_end_grad.positions,
                        "cell" => &mut system_end_grad.cell,
                        other => panic!("unsupported gradient parameter {}", other)
                    };
                    let system_start_grad = *system_end_grad;

                    let mut samples = LabelsBuilder::new(gradient.samples.names());
                    for gradient_sample in gradient.samples.iter().skip(system_start_grad) {
                        let sample_i = gradient_sample[0].usize();
                        let structure = structure_per_sample[sample_i];
                        if structure.usize() == system_i {
                            // this sample is part of to the current system
                            let mut new_gradient_sample = gradient_sample.to_vec();
                            new_gradient_sample[0] = samples_mapping[&sample_i].into();
                            samples.add(&new_gradient_sample);

                            *system_end_grad += 1;
                        } else if structure.usize() > system_i {
                            // found the next system
                            break;
                        } else {
                            // structure.usize() < system_i
                            panic!("expected samples to be ordered by system, they are not");
                        }
                    }

                    let mut shape = Vec::new();

                    let samples = Arc::new(samples.finish());
                    shape.push(samples.count());

                    let mut components = Vec::new();
                    for component in &gradient.components {
                        components.push(Arc::clone(component));
                        shape.push(component.count());
                    }

                    shape.push(n_properties);

                    let per_sample_size: usize = shape.iter().skip(1).product();
                    let data_ptr = unsafe {
                        // SAFETY: same as the values above, this is creating
                        // multiple non-overlapping regions in memory
                        gradient.data.as_array_mut().as_mut_ptr().add(per_sample_size * system_start_grad)
                    };

                    let data = UnsafeArrayViewMut {
                        shape: shape,
                        data: data_ptr,
                    };
                    new_block.add_gradient(parameter, data, samples, components).expect("invalid gradients");
                }

                return new_block;
            }).collect();

        let tensor = TensorMapView {
            data: TensorMap::new(descriptor.keys().clone(), blocks).expect("invalid TensorMap"),
            marker: std::marker::PhantomData
        };

        descriptor_by_system.push(tensor);
    }

    return descriptor_by_system;
}
