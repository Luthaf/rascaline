use std::os::raw::c_char;
use std::ffi::CStr;

use crate::Error;

use super::{catch_unwind, featomic_status_t};
use super::utils::copy_str_to_c;

/// Clear all collected profiling data
///
/// See also `featomic_profiling_enable` and `featomic_profiling_get`.
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub extern fn featomic_profiling_clear() -> featomic_status_t {
    catch_unwind(|| {
        time_graph::clear_collected_data();
        Ok(())
    })
}


/// Enable or disable profiling data collection. By default, data collection
/// is disabled.
///
/// Featomic uses the [`time_graph`](https://docs.rs/time-graph/) to collect
/// timing information on the calculations. This profiling code collects the
/// total time spent inside the most important functions, as well as the
/// function call graph (which function called which other function).
///
/// You can use `featomic_profiling_clear` to reset profiling data to an empty
/// state, and `featomic_profiling_get` to extract the profiling data.
///
/// @param enabled whether data collection should be enabled or not
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub extern fn featomic_profiling_enable(enabled: bool) -> featomic_status_t {
    catch_unwind(|| {
        time_graph::enable_data_collection(enabled);
        Ok(())
    })
}

/// Extract the current set of data collected for profiling.
///
/// See also `featomic_profiling_enable` and `featomic_profiling_clear`.
///
/// @param format in which format should the data be provided. `"table"`,
///              `"short_table"` and `"json"` are currently supported
/// @param buffer pre-allocated buffer in which profiling data will be copied.
///               If the buffer is too small, this function will return
///               `FEATOMIC_BUFFER_SIZE_ERROR`
/// @param bufflen size of the `buffer`
///
/// @returns The status code of this operation. If the status is not
///          `FEATOMIC_SUCCESS`, you can use `featomic_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn featomic_profiling_get(
    format: *const c_char,
    buffer: *mut c_char,
    bufflen: usize,
) -> featomic_status_t {
    catch_unwind(|| {
        check_pointers!(format);

        let data = match CStr::from_ptr(format).to_str()? {
            "table" => {
                time_graph::get_full_graph().as_table()
            },
            "short_table" => {
                time_graph::get_full_graph().as_short_table()
            },
            "json" => {
                time_graph::get_full_graph().as_json()
            },
            format => return Err(Error::InvalidParameter(format!(
                "invalid data format in featomic_profiling_get: {}, expected 'table', 'short_table' or 'json'",
                format
            )))
        };
        copy_str_to_c(&data, buffer, bufflen)?;

        Ok(())
    })
}
