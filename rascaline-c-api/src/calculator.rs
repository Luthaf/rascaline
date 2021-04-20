use std::os::raw::c_char;
//use std::ffi::{CStr, CString};
use std::ffi::{CStr};
use std::ops::{Deref, DerefMut};

use rascaline::{Calculator, System, CalculationOptions, SelectedIndexes};

use super::utils::copy_str_to_c;
//use super::{catch_unwind, rascal_status_t, GLOBAL_CALLBACK};
use super::{catch_unwind, rascal_status_t};

use super::descriptor::rascal_descriptor_t;
use super::system::rascal_system_t;

/// Opaque type representing a `Calculator`
#[allow(non_camel_case_types)]
pub struct rascal_calculator_t(Calculator);

impl Deref for rascal_calculator_t {
    type Target = Calculator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for rascal_calculator_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Create a new calculator with the given `name` and `parameters`.
///
/// @verbatim embed:rst:leading-asterisk
///
/// The list of available calculators and the corresponding parameters are in
/// the :ref:`main documentation <calculators-list>`. The ``parameters`` should
/// be formatted as JSON, according to the requested calculator schema.
///
/// @endverbatim
///
/// All memory allocated by this function can be released using
/// `rascal_calculator_free`.
///
/// @param name name of the calculator as a NULL-terminated string
/// @param parameters hyper-parameters of the calculator, JSON-formatted in a
///                   NULL-terminated string
///
/// @returns A pointer to the newly allocated calculator, or a `NULL` pointer in
///          case of error. In case of error, you can use `rascal_last_error()`
///          to get the error message.
#[no_mangle]
#[allow(clippy::module_name_repetitions)]
pub unsafe extern fn rascal_calculator(name: *const c_char, parameters: *const c_char) -> *mut rascal_calculator_t {
    let mut raw = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut raw);
    let status = catch_unwind(move || {
        check_pointers!(name, parameters);
        let name = CStr::from_ptr(name).to_str()?;
        let parameters = CStr::from_ptr(parameters).to_str()?;
        let calculator = Calculator::new(name, parameters.to_owned())?;
        let boxed = Box::new(rascal_calculator_t(calculator));

        *unwind_wrapper.0 = Box::into_raw(boxed);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return raw;
}

/// Free the memory associated with a `calculator` previously created with
/// `rascal_calculator`.
///
/// If `calculator` is `NULL`, this function does nothing.
///
/// @param calculator pointer to an existing calculator, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the
///          full error message.
#[no_mangle]
pub unsafe extern fn rascal_calculator_free(calculator: *mut rascal_calculator_t) -> rascal_status_t {
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
///`name` will be NULL-terminated by this function. If the buffer is too small
/// to fit the whole name, this function will return
/// `RASCAL_INVALID_PARAMETER_ERROR`
///
/// @param calculator pointer to an existing calculator
/// @param name string buffer to fill with the calculator name
/// @param bufflen number of characters available in the buffer
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_calculator_name(
    calculator: *const rascal_calculator_t,
    name: *mut c_char,
    bufflen: usize
) -> rascal_status_t {
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
/// `RASCAL_INVALID_PARAMETER_ERROR`.
///
/// @param calculator pointer to an existing calculator
/// @param parameters string buffer to fill with the parameters used to create
///                   this calculator
/// @param bufflen number of characters available in the buffer
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_calculator_parameters(
    calculator: *const rascal_calculator_t,
    parameters: *mut c_char,
    bufflen: usize
) -> rascal_status_t {
    catch_unwind(|| {
        check_pointers!(calculator, parameters);
        copy_str_to_c(&(*calculator).parameters(), parameters, bufflen)?;
        Ok(())
    })
}

/// Options that can be set to change how a calculator operates.
#[repr(C)]
pub struct rascal_calculation_options_t {
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    use_native_system: bool,
    /// List of samples on which to run the calculation. Use `NULL` to run the
    /// calculation on all samples. The samples must be represented as a
    /// row-major array, containing values similar to the samples index of a
    /// descriptor. If necessary, gradients samples will be derived from the
    /// values given in selected_samples.
    selected_samples: *const i32,
    /// If selected_samples is not `NULL`, this should be set to the size of the
    /// selected_samples array
    selected_samples_count: usize,
    /// List of features on which to run the calculation. Use `NULL` to run the
    /// calculation on all features. The features must be represented as a
    /// row-major array, containing values similar to the features index of a
    /// descriptor.
    selected_features: *const i32,
    /// If selected_features is not `NULL`, this should be set to the size of the
    /// selected_features array
    selected_features_count: usize,
}

impl<'a> From<&'a rascal_calculation_options_t> for CalculationOptions<'a> {
    fn from(options: &'a rascal_calculation_options_t) -> CalculationOptions {
        let selected_samples = if options.selected_samples.is_null() {
            SelectedIndexes::All
        } else {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    options.selected_samples.cast(),
                    options.selected_samples_count
                )
            };
            SelectedIndexes::FromC(slice)
        };

        let selected_features = if options.selected_features.is_null() {
            SelectedIndexes::All
        } else {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    options.selected_features.cast(),
                    options.selected_features_count
                )
            };
            SelectedIndexes::FromC(slice)
        };

        CalculationOptions {
            use_native_system: options.use_native_system,
            selected_samples: selected_samples,
            selected_features: selected_features,
        }
    }
}

#[allow(clippy::doc_markdown)]
/// Run a calculation with the given `calculator` on the given `systems`,
/// storing the resulting data in the `descriptor`.
///
/// @param calculator pointer to an existing calculator
/// @param descriptor pointer to an existing descriptor for data storage
/// @param systems pointer to an array of systems implementation
/// @param systems_count number of systems in `systems`
/// @param options options for this calculation
///
/// @returns The status code of this operation. If the status is not
///          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn rascal_calculator_compute(
    calculator: *mut rascal_calculator_t,
    descriptor: *mut rascal_descriptor_t,
    systems: *mut rascal_system_t,
    systems_count: usize,
    options: rascal_calculation_options_t,
) -> rascal_status_t {
    //RascalLogger.log(0, "test");
    // TODO(alex) make a proper test, and set a default logger somewhere
    //let to_print = CString::new("test").unwrap();
    //let to_print_ptr = to_print.as_ptr();
    ////(GLOBAL_CALLBACK.expect("No callback function was set."))(to_print_ptr);
    ////match *(GLOBAL_CALLBACK.lock().unwrap()) {
    ////    Some(p) => p(5, to_print_ptr),
    ////    None => println!("No callback function was set."),
    ////}
    //match *(GLOBAL_CALLBACK.lock().unwrap()) {
    //    Some(p) => p(0, to_print_ptr),
    //    None => println!("No callback function was set."),
    //}
    catch_unwind(|| {
        if systems_count == 0 {
            log::warn!("0 systems given to rascal_calculator_compute, we will do nothing");
            return Ok(());
        }
        check_pointers!(calculator, descriptor, systems);

        // Create a Vec<Box<dyn System>> from the passed systems
        let c_systems = std::slice::from_raw_parts_mut(systems, systems_count);
        let mut systems = Vec::with_capacity(c_systems.len());
        for system in c_systems {
            systems.push(Box::new(system) as Box<dyn System>);
        }

        let options = (&options).into();
        (*calculator).compute(&mut systems, &mut *descriptor, options)
    })
}
