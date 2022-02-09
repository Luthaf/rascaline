use std::ops::{Deref, DerefMut};
use std::os::raw::c_char;
use std::ffi::CStr;

use rascaline::descriptor::{Descriptor, IndexValue};
use rascaline::Error;
use super::{catch_unwind, rascal_status_t};

/// Opaque type representing a `Descriptor`.
#[allow(non_camel_case_types)]
pub struct rascal_descriptor_t(Descriptor);

impl Deref for rascal_descriptor_t {
    type Target = Descriptor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for rascal_descriptor_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Create a new empty descriptor.
///
/// All memory allocated by this function can be released using
/// `rascal_descriptor_free`.
///
/// @returns A pointer to the newly allocated descriptor, or a `NULL` pointer in
///          case of error. In case of error, you can use `rascal_last_error()`
///          to get the error message.
#[no_mangle]
#[allow(clippy::module_name_repetitions)]
pub unsafe extern fn rascal_descriptor() -> *mut rascal_descriptor_t {
    let descriptor = Box::new(rascal_descriptor_t(Descriptor::new()));
    return Box::into_raw(descriptor);
}

/// Free the memory associated with a `descriptor` previously created with
/// `rascal_descriptor`.
///
/// If `descriptor` is `NULL`, this function does nothing.
///
/// @param descriptor pointer to an existing descriptor, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the
///          full error message.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_free(descriptor: *mut rascal_descriptor_t) -> rascal_status_t {
    catch_unwind(|| {
        if !descriptor.is_null() {
            let boxed = Box::from_raw(descriptor);
            std::mem::drop(boxed);
        }
        Ok(())
    })
}

/// Get the values stored inside this descriptor after a call to
/// `rascal_calculator_compute`.
///
/// This function sets `*data` to a pointer containing the address of first
/// element of the 2D array containing the values, `*samples` to the size of the
/// first axis of this array and `*features` to the size of the second axis of
/// the array. The array is stored using a row-major layout.
///
/// @param descriptor pointer to an existing descriptor
/// @param data pointer to a pointer to a double, will be set to the address of
///             the first element in the values array
/// @param samples pointer to a single integer, will be set to the first
///                dimension of the values array
/// @param features pointer to a single integer, will be set to the second
///                 dimension of the values array
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_values(
    descriptor: *mut rascal_descriptor_t,
    data: *mut *mut f64,
    samples: *mut usize,
    features: *mut usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, data, samples, features);

        let array = &mut (*descriptor).values;
        if array.is_empty() {
            *data = std::ptr::null_mut();
        } else {
            *data = array.as_mut_ptr();
        }

        let shape = array.shape();
        *samples = shape[0];
        *features = shape[1];

        Ok(())
    })
}

#[allow(clippy::doc_markdown)]
/// Get the gradients stored inside this descriptor after a call to
/// `rascal_calculator_compute`, if any.
///
/// This function sets `*data` to to a pointer containing the address of the
/// first element of the 2D array containing the gradients, `*gradient_samples`
/// to the size of the first axis of this array and `*features` to the size of
/// the second axis of the array. The array is stored using a row-major layout.
///
/// If this descriptor does not contain gradient data, `*data` is set to `NULL`,
/// while `*gradient_samples` and `*features` are set to 0.
///
/// @param descriptor pointer to an existing descriptor
/// @param data pointer to a pointer to a double, will be set to the address of
///             the first element in the gradients array
/// @param gradient_samples pointer to a single integer, will be set to the first
///                         dimension of the gradients array
/// @param features pointer to a single integer, will be set to the second
///                 dimension of the gradients array
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_gradients(
    descriptor: *mut rascal_descriptor_t,
    data: *mut *mut f64,
    gradient_samples: *mut usize,
    features: *mut usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, data, gradient_samples, features);

        if let Some(ref mut array) = (*descriptor).gradients {
            *data = array.as_mut_ptr();
            let shape = array.shape();
            *gradient_samples = shape[0];
            *features = shape[1];
        } else {
            *data = std::ptr::null_mut();
            *gradient_samples = 0;
            *features = 0;
        }

        Ok(())
    })
}

#[repr(C)]
#[allow(non_camel_case_types)]
/// The different kinds of indexes that can exist on a `rascal_descriptor_t`
pub enum rascal_indexes_kind {
    /// The feature index, describing the features of the representation
    RASCAL_INDEXES_FEATURES = 0,
    /// The samples index, describing different samples in the representation
    RASCAL_INDEXES_SAMPLES = 1,
    /// The gradient samples index, describing the gradients of samples in the
    /// representation with respect to other atoms
    RASCAL_INDEXES_GRADIENT_SAMPLES = 2,
}

/// Indexes representing metadata associated with either samples or features in
/// a given descriptor.
#[repr(C)]
pub struct rascal_indexes_t {
    /// Names of the variables composing this set of indexes. There are `size`
    /// elements in this array, each being a NULL terminated string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different variables in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of variables/size of a single entry in the set of indexes
    pub size: usize,
    /// Number entries in the set of indexes
    pub count: usize,
}

/// Get the values associated with one of the `indexes` in the given
/// `descriptor`.
///
/// This function sets `indexes->names` to to a **read only** array containing
/// the names of the variables in this set of indexes; `indexes->values` to to a
/// **read only** 2D array containing values taken by these variables,
/// `indexes->count` to the number of indexes (first dimension of the array) and
/// `indexes->values` to the size of each index (second dimension of the array).
/// The array is stored using a row-major layout.
///
/// If this `descriptor` does not contain gradient data, and `indexes` is
/// `RASCAL_INDEXES_GRADIENTS`, all members of `indexes` are set to `NULL` or 0.
///
/// @param descriptor pointer to an existing descriptor
/// @param kind type of indexes requested
/// @param indexes pointer to `rascal_indexes_t` that will be filled by this function
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_indexes(
    descriptor: *const rascal_descriptor_t,
    kind: rascal_indexes_kind,
    indexes: *mut rascal_indexes_t,
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, indexes);

        let rust_indexes = match kind {
            rascal_indexes_kind::RASCAL_INDEXES_FEATURES => &(*descriptor).features,
            rascal_indexes_kind::RASCAL_INDEXES_SAMPLES => &(*descriptor).samples,
            rascal_indexes_kind::RASCAL_INDEXES_GRADIENT_SAMPLES => {
                if let Some(indexes) = &(*descriptor).gradients_samples {
                    indexes
                } else {
                    (*indexes).values = std::ptr::null();
                    (*indexes).names = std::ptr::null();
                    (*indexes).size = 0;
                    (*indexes).count = 0;
                    return Ok(());
                }
            }
        };

        (*indexes).size = rust_indexes.size();
        (*indexes).count = rust_indexes.count();

        if rust_indexes.count() == 0 {
            (*indexes).values = std::ptr::null();
        } else {
            (*indexes).values = (&rust_indexes[0][0] as *const IndexValue).cast();
        }

        if rust_indexes.size() == 0 {
            (*indexes).names = std::ptr::null();
        } else {
            (*indexes).names = rust_indexes.c_names().as_ptr().cast();
        }

        Ok(())
    })
}

/// Make the given `descriptor` dense along the given `variables`.
///
/// The `variable` array should contain the name of the variables as
/// NULL-terminated strings, and `variables_count` must be the number of
/// variables in the array.
///
/// The `requested` parameter defines which set of values taken by the
/// `variables` should be part of the new features. If it is `NULL`, this is the
/// set of values taken by the variables in the samples. Otherwise, it must be a
/// pointer to the first element of a 2D row-major array with one row for each
/// new feature block, and one column for each variable. `requested_size` must
/// be the number of rows in this array.
///
/// This function "moves" the variables from the samples to the features,
/// filling the new features with zeros if the corresponding sample is missing.
///
/// For example, take a descriptor containing two samples variables (`structure`
/// and `species`) and two features (`n` and `l`). Starting with this
/// descriptor:
///
/// ```text
///                       +---+---+---+
///                       | n | 0 | 1 |
///                       +---+---+---+
///                       | l | 0 | 1 |
/// +-----------+---------+===+===+===+
/// | structure | species |           |
/// +===========+=========+   +---+---+
/// |     0     |    1    |   | 1 | 2 |
/// +-----------+---------+   +---+---+
/// |     0     |    6    |   | 3 | 4 |
/// +-----------+---------+   +---+---+
/// |     1     |    6    |   | 5 | 6 |
/// +-----------+---------+   +---+---+
/// |     1     |    8    |   | 7 | 8 |
/// +-----------+---------+---+---+---+
/// ```
///
/// Calling `descriptor.densify({"species"})` will move `species` out of the
/// samples and into the features, producing:
/// ```text
///             +---------+-------+-------+-------+
///             | species |   1   |   6   |   8   |
///             +---------+---+---+---+---+---+---+
///             |    n    | 0 | 1 | 0 | 1 | 0 | 1 |
///             +---------+---+---+---+---+---+---+
///             |    l    | 0 | 1 | 0 | 1 | 0 | 1 |
/// +-----------+=========+===+===+===+===+===+===+
/// | structure |
/// +===========+         +---+---+---+---+---+---+
/// |     0     |         | 1 | 2 | 3 | 4 | 0 | 0 |
/// +-----------+         +---+---+---+---+---+---+
/// |     1     |         | 0 | 0 | 5 | 6 | 7 | 8 |
/// +-----------+---------+---+---+---+---+---+---+
/// ```
///
/// Notice how there is only one row/sample for each structure now, and how each
/// value for `species` have created a full block of features. Missing values
/// (e.g. structure 0/species 8) have been filled with 0.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_densify(
    descriptor: *mut rascal_descriptor_t,
    variables: *const *const c_char,
    variables_count: usize,
    requested: *const i32,
    requested_size: usize,
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variables);
        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str()?;
            rust_variables.push(variable);
        }

        let requested = if requested.is_null() {
            None
        } else {
            Some(ndarray::ArrayView2::from_shape_ptr(
                [requested_size, variables_count], requested.cast::<IndexValue>()
            ))
        };

        (*descriptor).densify(&rust_variables, requested)?;

        Ok(())
    })
}

/// `rascal_densified_position_t` contains all the information to reconstruct
/// the new position of the values associated with a single sample in the
/// initial descriptor after a call to `rascal_descriptor_densify_values`
#[repr(C)]
pub struct rascal_densified_position_t {
    /// if `used` is `true`, index of the new sample in the value array
    pub new_sample: usize,
    /// if `used` is `true`, index of the feature block in the new array
    pub feature_block: usize,
    /// indicate whether this sample was needed to construct the new value
    /// array. This might be `false` when the value of densified variables
    /// specified by the user does not match the sample.
    pub used: bool,
}

/// Make this descriptor dense along the given `variables`, only modifying the
/// values array, and not the gradients array.
///
/// This function behaves similarly to `rascal_descriptor_densify`, please refer
/// to its documentation for more information.
///
/// If this descriptor contains gradients, `densified_positions` will point to
/// an array allocated with `malloc` containing the changes made to the values,
/// which can be used to reconstruct the change to make to the gradients. The
/// size of this array will be stored in `densified_positions_count`.
///
/// Users of this function are expected to `free` the corresponding memory when
/// they no longer need it.
///
/// This is an advanced function most users should not need to use, used to
/// implement backward propagation without having to densify the full gradient
/// array.
#[no_mangle]
pub unsafe extern fn rascal_descriptor_densify_values(
    descriptor: *mut rascal_descriptor_t,
    variables: *const *const c_char,
    variables_count: usize,
    requested: *const i32,
    requested_size: usize,
    densified_positions: *mut *mut rascal_densified_position_t,
    densified_positions_count: *mut usize,
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variables);
        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str()?;
            rust_variables.push(variable);
        }

        let requested = if requested.is_null() {
            None
        } else {
            Some(ndarray::ArrayView2::from_shape_ptr(
                [requested_size, variables_count], requested.cast::<IndexValue>()
            ))
        };

        let densified_positions_rust = (*descriptor).densify_values(&rust_variables, requested)?;
        if densified_positions_rust.is_empty() {
            *densified_positions_count = 0;
            *densified_positions = std::ptr::null_mut();
            return Ok(());
        }

        *densified_positions_count = densified_positions_rust.len();
        *densified_positions = libc::calloc(
            densified_positions_rust.len(),
            std::mem::size_of::<rascal_densified_position_t>()
        ).cast();

        if (*densified_positions).is_null() {
            return Err(Error::BufferSize(
                "failed to allocate enough memory to store densified positions".into()
            ));
        }

        for (old_sample_i, position) in densified_positions_rust.iter().enumerate() {
            if let Some(position) = position {
                (*(*densified_positions).add(old_sample_i)).new_sample = position.sample;
                (*(*densified_positions).add(old_sample_i)).feature_block = position.features_block;
                (*(*densified_positions).add(old_sample_i)).used = true;
            } else {
                (*(*densified_positions).add(old_sample_i)).new_sample = 0;
                (*(*densified_positions).add(old_sample_i)).feature_block = 0;
                (*(*densified_positions).add(old_sample_i)).used = false;
            }
        }

        Ok(())
    })
}
