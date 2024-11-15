use std::os::raw::c_char;
use std::ffi::CStr;
use std::ops::{Deref, DerefMut};

use metatensor::{Labels, TensorMap};
use metatensor::c_api::{mts_tensormap_t, mts_labels_t};
use featomic::{Calculator, System, CalculationOptions, LabelsSelection};

use super::utils::copy_str_to_c;
use super::{catch_unwind, featomic_status_t};

use super::system::featomic_system_t;

/// Opaque type representing a `Calculator`
#[allow(non_camel_case_types)]
pub struct featomic_calculator_t(Calculator);

impl Deref for featomic_calculator_t {
    type Target = Calculator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for featomic_calculator_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Create a new calculator with the given `name` and `parameters`.
///
/// @verbatim embed:rst:leading-asterisk
///
/// The list of available calculators and the corresponding parameters are in
/// the :ref:`main documentation <userdoc-references>`. The ``parameters`` should
/// be formatted as JSON, according to the requested calculator schema.
///
/// @endverbatim
///
/// All memory allocated by this function can be released using
/// `featomic_calculator_free`.
///
/// @param name name of the calculator as a NULL-terminated string
/// @param parameters hyper-parameters of the calculator, JSON-formatted in a
///                   NULL-terminated string
///
/// @returns A pointer to the newly allocated calculator, or a `NULL` pointer in
///          case of error. In case of error, you can use `featomic_last_error()`
///          to get the error message.
#[no_mangle]
#[allow(clippy::module_name_repetitions)]
pub unsafe extern fn featomic_calculator(name: *const c_char, parameters: *const c_char) -> *mut featomic_calculator_t {
    let mut raw = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut raw);
    let status = catch_unwind(move || {
        check_pointers!(name, parameters);
        let name = CStr::from_ptr(name).to_str()?;
        let parameters = CStr::from_ptr(parameters).to_str()?;
        let calculator = Calculator::new(name, parameters.to_owned())?;
        let boxed = Box::new(featomic_calculator_t(calculator));

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = Box::into_raw(boxed);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return raw;
}

/// Free the memory associated with a `calculator` previously created with
/// `featomic_calculator`.
///
/// If `calculator` is `NULL`, this function does nothing.
///
/// @param calculator pointer to an existing calculator, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the
///          full error message.
#[no_mangle]
pub unsafe extern fn featomic_calculator_free(calculator: *mut featomic_calculator_t) -> featomic_status_t {
    catch_unwind(|| {
        if !calculator.is_null() {
            let boxed = Box::from_raw(calculator);
            std::mem::drop(boxed);
        }

        Ok(())
    })
}

/// Get a copy of the name of this calculator in the `name` buffer of size
/// `bufflen`.
///
/// `name` will be NULL-terminated by this function. If the buffer is too small
/// to fit the whole name, this function will return
/// `FEATOMIC_BUFFER_SIZE_ERROR`
///
/// @param calculator pointer to an existing calculator
/// @param name string buffer to fill with the calculator name
/// @param bufflen number of characters available in the buffer
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn featomic_calculator_name(
    calculator: *const featomic_calculator_t,
    name: *mut c_char,
    bufflen: usize
) -> featomic_status_t {
    catch_unwind(|| {
        check_pointers!(calculator, name);
        copy_str_to_c(&(*calculator).name(), name, bufflen)?;
        Ok(())
    })
}

/// Get a copy of the parameters used to create this calculator in the
/// `parameters` buffer of size `bufflen`.
///
/// `parameters` will be NULL-terminated by this function. If the buffer is too
/// small to fit the whole name, this function will return
/// `FEATOMIC_BUFFER_SIZE_ERROR`.
///
/// @param calculator pointer to an existing calculator
/// @param parameters string buffer to fill with the parameters used to create
///                   this calculator
/// @param bufflen number of characters available in the buffer
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn featomic_calculator_parameters(
    calculator: *const featomic_calculator_t,
    parameters: *mut c_char,
    bufflen: usize
) -> featomic_status_t {
    catch_unwind(|| {
        check_pointers!(calculator, parameters);
        copy_str_to_c((*calculator).parameters(), parameters, bufflen)?;
        Ok(())
    })
}


#[allow(clippy::doc_markdown)]
/// Get all radial cutoffs used by this `calculator`'s neighbors lists (which
/// can be an empty list).
///
/// The `*cutoffs` pointer will be pointing to data inside the `calculator`, and
/// is only valid when the `calculator` itself is.
///
/// @param calculator pointer to an existing calculator
/// @param cutoffs pointer to be filled with the address of the first element of
///                an array of cutoffs
/// @param cutoffs_count pointer to be filled with the number of elements in the
///                      `cutoffs` array
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn featomic_calculator_cutoffs(
    calculator: *const featomic_calculator_t,
    cutoffs: *mut *const f64,
    cutoffs_count: *mut usize
) -> featomic_status_t {
    catch_unwind(|| {
        check_pointers!(calculator, cutoffs, cutoffs_count);

        let slice = (*calculator).cutoffs();
        *cutoffs = slice.as_ptr();
        *cutoffs_count = slice.len();

        Ok(())
    })
}

/// Rules to select labels (either samples or properties) on which the user
/// wants to run a calculation
///
/// To run the calculation for all possible labels, users should set both fields
/// to NULL.
#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct featomic_labels_selection_t {
    /// Select a subset of labels, using the same selection criterion for all
    /// keys in the final `mts_tensormap_t`.
    ///
    /// If the `mts_labels_t` instance contains the same variables as the full
    /// set of labels, then only entries from the full set that also appear in
    /// this selection will be used.
    ///
    /// If the `mts_labels_t` instance contains a subset of the variables of the
    /// full set of labels, then only entries from the full set which match one
    /// of the entry in this selection for all of the selection variable will be
    /// used.
    subset: *const mts_labels_t,
    /// Use a predefined subset of labels, with different entries for different
    /// keys of the final `mts_tensormap_t`.
    ///
    /// For each key, the corresponding labels are fetched out of the
    /// `mts_tensormap_t` instance, which must have the same set of keys as the
    /// full calculation.
    predefined: *const mts_tensormap_t,
}

fn c_labels_to_rust(mut labels: mts_labels_t) -> Result<mts_labels_t, featomic::Error> {
    if labels.internal_ptr_.is_null() {
        // create new metatensor-core labels
        unsafe {
            metatensor::errors::check_status(
                metatensor::c_api::mts_labels_create(&mut labels)
            )?;
        }

        return Ok(labels);
    } else {
        // increment reference count
        let mut clone = mts_labels_t {
            internal_ptr_: std::ptr::null_mut(),
            names: std::ptr::null(),
            values: std::ptr::null(),
            size: 0,
            count: 0
        };
        unsafe {
            metatensor::errors::check_status(
                metatensor::c_api::mts_labels_clone(labels, &mut clone)
            )?;
        }
        return Ok(clone);
    }
}

fn convert_labels_selection<'a>(
    selection: &'a featomic_labels_selection_t,
    labels: &'a mut Option<Labels>,
    predefined: &'a mut Option<TensorMap>,
) -> Result<LabelsSelection<'a>, featomic::Error> {
    match (selection.subset.is_null(), selection.predefined.is_null()) {
        (true, true) => Ok(LabelsSelection::All),
        (false, true) => {
            *labels = unsafe {
                let raw_labels = c_labels_to_rust(*selection.subset)?;
                Some(Labels::from_raw(raw_labels))
            };

            Ok(LabelsSelection::Subset(labels.as_ref().expect("just created it")))
        }
        (true, false) => {
            let tensor = unsafe {
                TensorMap::from_raw(selection.predefined.cast_mut())
            };

            match tensor.try_clone() {
                Ok(copy) => {
                    // we don't own the `tensor`, so we should not run Drop on it
                    let _ = TensorMap::into_raw(tensor);
                    *predefined = Some(copy);
                }
                Err(e) => {
                    // same as above
                    let _ = TensorMap::into_raw(tensor);
                    return Err(featomic::Error::from(e));
                }
            }

            Ok(LabelsSelection::Predefined(predefined.as_ref().expect("just created it")))
        }
        (false, false) => {
            Err(featomic::Error::InvalidParameter(
                "can not have both global and predefined non-NULL in featomic_labels_selection_t".into()
            ))
        }
    }
}

fn key_selection(value: *const mts_labels_t, labels: &'_ mut Option<Labels>) -> Result<Option<&'_ Labels>, featomic::Error> {
    if value.is_null() {
        return Ok(None);
    }

    unsafe {
        let raw_labels = c_labels_to_rust(*value)?;
        *labels = Some(Labels::from_raw(raw_labels));
    }

    return Ok(labels.as_ref());
}

/// Options that can be set to change how a calculator operates.
#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct featomic_calculation_options_t {
    /// @verbatim embed:rst:leading-asterisk
    /// Array of NULL-terminated strings containing the gradients to compute.
    /// If this field is `NULL` and `gradients_count` is 0, no gradients are
    /// computed.
    ///
    /// The following gradients are available:
    ///
    /// - ``"positions"``, for gradients of the representation with respect to
    ///   atomic positions, with fixed cell matrix parameters. Positions
    ///   gradients are computed as
    ///
    ///   .. math::
    ///       \frac{\partial \langle q \vert A_i \rangle}
    ///            {\partial \mathbf{r_j}}
    ///
    ///   where :math:`\langle q \vert A_i \rangle` is the representation around
    ///   atom :math:`i` and :math:`\mathbf{r_j}` is the position vector of the
    ///   atom :math:`j`.
    ///
    ///   **Note**: Position gradients of an atom are computed with respect to all
    ///   other atoms within the representation. To recover the force one has to
    ///   accumulate all pairs associated with atom :math:`i`.
    ///
    /// - ``"strain"``, for gradients of the representation with respect to
    ///   strain. These gradients are typically used to compute the virial, and
    ///   from there the pressure acting on a system. To compute them, we
    ///   pretend that all the positions :math:`\mathbf r` and unit cell
    ///   :math:`\mathbf H` have been scaled by a strain matrix
    ///   :math:`\epsilon`:
    ///
    ///   .. math::
    ///      \mathbf r &\rightarrow \mathbf r \left(\mathbb{1} + \epsilon \right)\\
    ///      \mathbf H &\rightarrow \mathbf H \left(\mathbb{1} + \epsilon \right)
    ///
    ///   and then take the gradients of the representation with respect to this
    ///   matrix:
    ///
    ///   .. math::
    ///       \frac{\partial \langle q \vert A_i \rangle} {\partial \mathbf{\epsilon}}
    ///
    /// - ``"cell"``, for gradients of the representation with respect to the
    ///   system's cell parameters. These gradients are computed at fixed
    ///   positions, and often not what you want when computing gradients
    ///   explicitly (they are mainly used in ``featomic.torch`` to integrate
    ///   with backward propagation). If you are trying to compute the virial
    ///   or the stress, you should use ``"strain"`` gradients instead.
    ///
    ///   .. math::
    ///       \left. \frac{\partial \langle q \vert A_i \rangle}
    ///            {\partial \mathbf{H}} \right |_\mathbf{r}
    ///
    /// @endverbatim
    gradients: *const *const c_char,
    /// Size of the `gradients` array
    gradients_count: usize,
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    use_native_system: bool,
    /// Selection of samples on which to run the computation
    selected_samples: featomic_labels_selection_t,
    /// Selection of properties to compute for the samples
    selected_properties: featomic_labels_selection_t,
    /// Selection for the keys to include in the output. Set this parameter to
    /// `NULL` to use the default set of keys, as determined by the calculator.
    /// Note that this default set of keys can depend on which systems we are
    /// running the calculation on.
    selected_keys: *const mts_labels_t,
}

#[allow(clippy::doc_markdown)]
/// Compute the representation of the given list of `systems` with a
/// `calculator`
///
/// This function allocates a new `mts_tensormap_t` in `*descriptor`, which
/// memory needs to be released by the user with `mts_tensormap_free`.
///
/// @param calculator pointer to an existing calculator
/// @param descriptor pointer to an `mts_tensormap_t *` that will be allocated
///                   by this function
/// @param systems pointer to an array of systems implementation
/// @param systems_count number of systems in `systems`
/// @param options options for this calculation
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn featomic_calculator_compute(
    calculator: *mut featomic_calculator_t,
    descriptor: *mut *mut mts_tensormap_t,
    systems: *mut featomic_system_t,
    systems_count: usize,
    options: featomic_calculation_options_t,
) -> featomic_status_t {
    catch_unwind(move || {
        if systems_count == 0 {
            log::warn!("0 systems given to featomic_calculator_compute, nothing to do");
            return Ok(());
        }
        check_pointers!(calculator, descriptor, systems);

        // Create a Vec<Box<dyn System>> from the passed systems
        let c_systems = if systems_count == 0 {
            &mut []
        } else {
            assert_ne!(systems, std::ptr::null_mut());
            std::slice::from_raw_parts_mut(systems, systems_count)
        };
        let mut systems = Vec::with_capacity(c_systems.len());
        for system in c_systems {
            systems.push(Box::new(system) as Box<dyn System>);
        }

        let c_gradients = if options.gradients_count == 0 {
            &[]
        } else {
            assert_ne!(options.gradients, std::ptr::null());
            std::slice::from_raw_parts(options.gradients, options.gradients_count)
        };
        let mut gradients = Vec::new();
        for &parameter in c_gradients {
            gradients.push(CStr::from_ptr(parameter).to_str()?);
        }

        let mut selected_samples = None;
        let mut predefined_samples = None;
        let selected_samples = convert_labels_selection(
            &options.selected_samples,
            &mut selected_samples,
            &mut predefined_samples
        )?;

        let mut selected_properties = None;
        let mut predefined_properties = None;
        let selected_properties = convert_labels_selection(
            &options.selected_properties,
            &mut selected_properties,
            &mut predefined_properties
        )?;

        let mut selected_keys = None;
        let selected_keys = key_selection(options.selected_keys, &mut selected_keys)?;

        let rust_options = CalculationOptions {
            gradients: &gradients,
            use_native_system: options.use_native_system,
            selected_samples,
            selected_properties,
            selected_keys,
        };

        let tensor = (*calculator).compute(&mut systems, rust_options)?;

        *descriptor = TensorMap::into_raw(tensor);
        Ok(())
    })
}
