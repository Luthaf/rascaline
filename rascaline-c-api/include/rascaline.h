/* ============    Automatically generated file, DOT NOT EDIT.    ============ *
 *                                                                             *
 *    This file is automatically generated from the rascaline-c-api sources,   *
 *    using cbindgen. If you want to make change to this file (including       *
 *    documentation), make the corresponding changes in the rust sources.      *
 * =========================================================================== */

#ifndef RASCALINE_H
#define RASCALINE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "equistore.h"

/**
 * Status code used when a function succeeded
 */
#define RASCAL_SUCCESS 0

/**
 * Status code used when a function got an invalid parameter
 */
#define RASCAL_INVALID_PARAMETER_ERROR 1

/**
 * Status code used when there was an error reading or writing JSON
 */
#define RASCAL_JSON_ERROR 2

/**
 * Status code used when a string contains non-utf8 data
 */
#define RASCAL_UTF8_ERROR 3

/**
 * Status code used for error related to reading files with chemfiles
 */
#define RASCAL_CHEMFILES_ERROR 4

/**
 * Status code used for errors coming from the system implementation if we
 * don't have a more specific status
 */
#define RASCAL_SYSTEM_ERROR 128

/**
 * Status code used when a memory buffer is too small to fit the requested data
 */
#define RASCAL_BUFFER_SIZE_ERROR 254

/**
 * Status code used when there was an internal error, i.e. there is a bug
 * inside rascaline
 */
#define RASCAL_INTERNAL_ERROR 255

/**
 * The "error" level designates very serious errors
 */
#define RASCAL_LOG_LEVEL_ERROR 1

/**
 * The "warn" level designates hazardous situations
 */
#define RASCAL_LOG_LEVEL_WARN 2

/**
 * The "info" level designates useful information
 */
#define RASCAL_LOG_LEVEL_INFO 3

/**
 * The "debug" level designates lower priority information
 *
 * By default, log messages at this level are disabled in release mode, and
 * enabled in debug mode.
 */
#define RASCAL_LOG_LEVEL_DEBUG 4

/**
 * The "trace" level designates very low priority, often extremely verbose,
 * information.
 *
 * By default, rascaline disable this level, you can enable it by editing the
 * code.
 */
#define RASCAL_LOG_LEVEL_TRACE 5

/**
 * Opaque type representing a `Calculator`
 */
typedef struct rascal_calculator_t rascal_calculator_t;

/**
 * Status type returned by all functions in the C API.
 *
 * The value 0 (`RASCAL_SUCCESS`) is used to indicate successful operations.
 * Positive non-zero values are reserved for internal use in rascaline.
 * Negative values are reserved for use in user code, in particular to indicate
 * error coming from callbacks.
 */
typedef int32_t rascal_status_t;

/**
 * Callback function type for rascaline logging system. Such functions are
 * called when a log event is emitted in the code.
 *
 * The first argument is the log level, one of `RASCAL_LOG_LEVEL_ERROR`,
 * `RASCAL_LOG_LEVEL_WARN` `RASCAL_LOG_LEVEL_INFO`, `RASCAL_LOG_LEVEL_DEBUG`,
 * or `RASCAL_LOG_LEVEL_TRACE`. The second argument is a NULL-terminated string
 * containing the message associated with the log event.
 */
typedef void (*rascal_logging_callback_t)(int32_t level, const char *message);

/**
 * Pair of atoms coming from a neighbor list
 */
typedef struct rascal_pair_t {
  /**
   * index of the first atom in the pair
   */
  uintptr_t first;
  /**
   * index of the second atom in the pair
   */
  uintptr_t second;
  /**
   * distance between the two atoms
   */
  double distance;
  /**
   * vector from the first atom to the second atom, wrapped inside the unit
   * cell as required by periodic boundary conditions.
   */
  double vector[3];
} rascal_pair_t;

/**
 * A `rascal_system_t` deals with the storage of atoms and related information,
 * as well as the computation of neighbor lists.
 *
 * This struct contains a manual implementation of a virtual table, allowing to
 * implement the rust `System` trait in C and other languages. Speaking in Rust
 * terms, `user_data` contains a pointer (analog to `Box<Self>`) to the struct
 * implementing the `System` trait; and then there is one function pointers
 * (`Option<unsafe extern fn(XXX)>`) for each function in the `System` trait.
 *
 * The `rascal_status_t` return value for the function is used to communicate
 * error messages. It should be 0/`RASCAL_SUCCESS` in case of success, any
 * non-zero value in case of error. The error will be propagated to the
 * top-level caller as a `RASCAL_SYSTEM_ERROR`
 *
 * A new implementation of the System trait can then be created in any language
 * supporting a C API (meaning any language for our purposes); by correctly
 * setting `user_data` to the actual data storage, and setting all function
 * pointers to the correct functions. For an example of code doing this, see
 * the `SystemBase` class in the Python interface to rascaline.
 *
 * **WARNING**: all function implementations **MUST** be thread-safe, function
 * taking `const` pointer parameters can be called from multiple threads at the
 * same time. The `rascal_system_t` itself might be moved from one thread to
 * another.
 */
typedef struct rascal_system_t {
  /**
   * User-provided data should be stored here, it will be passed as the
   * first parameter to all function pointers below.
   */
  void *user_data;
  /**
   * This function should set `*size` to the number of atoms in this system
   */
  rascal_status_t (*size)(const void *user_data, uintptr_t *size);
  /**
   * This function should set `*species` to a pointer to the first element of
   * a contiguous array containing the atomic species of each atom in the
   * system. Different atomic species should be identified with a different
   * value. These values are usually the atomic number, but don't have to be.
   * The array should contain `rascal_system_t::size()` elements.
   */
  rascal_status_t (*species)(const void *user_data, const int32_t **species);
  /**
   * This function should set `*positions` to a pointer to the first element
   * of a contiguous array containing the atomic cartesian coordinates.
   * `positions[0], positions[1], positions[2]` must contain the x, y, z
   * cartesian coordinates of the first atom, and so on.
   */
  rascal_status_t (*positions)(const void *user_data, const double **positions);
  /**
   * This function should write the unit cell matrix in `cell`, which have
   * space for 9 values. The cell should be written in row major order, i.e.
   * `ax ay az bx by bz cx cy cz`, where a/b/c are the unit cell vectors.
   */
  rascal_status_t (*cell)(const void *user_data, double *cell);
  /**
   * This function should compute the neighbor list with the given cutoff,
   * and store it for later access using `pairs` or `pairs_containing`.
   */
  rascal_status_t (*compute_neighbors)(void *user_data, double cutoff);
  /**
   * This function should set `*pairs` to a pointer to the first element of a
   * contiguous array containing all pairs in this system; and `*count` to
   * the size of the array/the number of pairs.
   *
   * This list of pair should only contain each pair once (and not twice as
   * `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
   * contains pairs where the distance between atoms is actually bellow the
   * cutoff passed in the last call to `compute_neighbors`. This function is
   * only valid to call after a call to `compute_neighbors`.
   */
  rascal_status_t (*pairs)(const void *user_data, const struct rascal_pair_t **pairs, uintptr_t *count);
  /**
   * This function should set `*pairs` to a pointer to the first element of a
   * contiguous array containing all pairs in this system containing the atom
   * with index `center`; and `*count` to the size of the array/the number of
   * pairs.
   *
   * The same restrictions on the list of pairs as `rascal_system_t::pairs`
   * applies, with the additional condition that the pair `i-j` should be
   * included both in the return of `pairs_containing(i)` and
   * `pairs_containing(j)`.
   */
  rascal_status_t (*pairs_containing)(const void *user_data, uintptr_t center, const struct rascal_pair_t **pairs, uintptr_t *count);
} rascal_system_t;

/**
 * Rules to select labels (either samples or properties) on which the user
 * wants to run a calculation
 *
 * To run the calculation for all possible labels, users should set both fields
 * to NULL.
 */
typedef struct rascal_labels_selection_t {
  /**
   * Select a subset of labels, using the same selection criterion for all
   * keys in the final `eqs_tensormap_t`.
   *
   * If the `eqs_labels_t` instance contains the same variables as the full
   * set of labels, then only entries from the full set that also appear in
   * this selection will be used.
   *
   * If the `eqs_labels_t` instance contains a subset of the variables of the
   * full set of labels, then only entries from the full set which match one
   * of the entry in this selection for all of the selection variable will be
   * used.
   */
  const eqs_labels_t *subset;
  /**
   * Use a predefined subset of labels, with different entries for different
   * keys of the final `eqs_tensormap_t`.
   *
   * For each key, the corresponding labels are fetched out of the
   * `eqs_tensormap_t` instance, which must have the same set of keys as the
   * full calculation.
   */
  const eqs_tensormap_t *predefined;
} rascal_labels_selection_t;

/**
 * Options that can be set to change how a calculator operates.
 */
typedef struct rascal_calculation_options_t {
  /**
   * Compute the gradients of the representation with respect to the atomic
   * positions, if they are implemented for this calculator
   */
  bool positions_gradient;
  /**
   * Compute the gradients of the representation with respect to the cell
   * vectors, if they are implemented for this calculator
   */
  bool cell_gradient;
  /**
   * Copy the data from systems into native `SimpleSystem`. This can be
   * faster than having to cross the FFI boundary too often.
   */
  bool use_native_system;
  /**
   * Selection of samples on which to run the computation
   */
  struct rascal_labels_selection_t selected_samples;
  /**
   * Selection of properties to compute for the samples
   */
  struct rascal_labels_selection_t selected_properties;
} rascal_calculation_options_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Get the last error message that was created on the current thread.
 *
 * @returns the last error message, as a NULL-terminated string
 */
const char *rascal_last_error(void);

/**
 * Set the given ``callback`` function as the global logging callback. This
 * function will be called on all log events. If a logging callback was already
 * set, it is replaced by the new one.
 */
rascal_status_t rascal_set_logging_callback(rascal_logging_callback_t callback);

/**
 * Read all structures in the file at the given `path` using
 * [chemfiles](https://chemfiles.org/), and convert them to an array of
 * `rascal_system_t`.
 *
 * This function can read all [formats supported by
 * chemfiles](https://chemfiles.org/chemfiles/latest/formats.html).
 *
 * This function allocates memory, which must be released using
 * `rascal_basic_systems_free`.
 *
 * If you need more control over the system behavior, consider writing your own
 * instance of `rascal_system_t`.
 *
 * @param path path of the file to read from in the local filesystem
 * @param systems `*systems` will be set to a pointer to the first element of
 *                 the array of `rascal_system_t`
 * @param count `*count` will be set to the number of systems read from the file
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_basic_systems_read(const char *path,
                                          struct rascal_system_t **systems,
                                          uintptr_t *count);

/**
 * Release memory allocated by `rascal_basic_systems_read`.
 *
 * This function is only valid to call with a pointer to systems obtained from
 * `rascal_basic_systems_read`, and the corresponding `count`. Any other use
 * will probably result in segmentation faults or double free. If `systems` is
 * NULL, this function does nothing.
 *
 * @param systems pointer to the first element of the array of
 * `rascal_system_t` @param count number of systems in the array
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_basic_systems_free(struct rascal_system_t *systems, uintptr_t count);

/**
 * Create a new calculator with the given `name` and `parameters`.
 *
 * @verbatim embed:rst:leading-asterisk
 *
 * The list of available calculators and the corresponding parameters are in
 * the :ref:`main documentation <calculators-list>`. The ``parameters`` should
 * be formatted as JSON, according to the requested calculator schema.
 *
 * @endverbatim
 *
 * All memory allocated by this function can be released using
 * `rascal_calculator_free`.
 *
 * @param name name of the calculator as a NULL-terminated string
 * @param parameters hyper-parameters of the calculator, JSON-formatted in a
 *                   NULL-terminated string
 *
 * @returns A pointer to the newly allocated calculator, or a `NULL` pointer in
 *          case of error. In case of error, you can use `rascal_last_error()`
 *          to get the error message.
 */
struct rascal_calculator_t *rascal_calculator(const char *name, const char *parameters);

/**
 * Free the memory associated with a `calculator` previously created with
 * `rascal_calculator`.
 *
 * If `calculator` is `NULL`, this function does nothing.
 *
 * @param calculator pointer to an existing calculator, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the
 *          full error message.
 */
rascal_status_t rascal_calculator_free(struct rascal_calculator_t *calculator);

/**
 * Get a copy of the name of this calculator in the `name` buffer of size
 * `bufflen`.
 *
 * `name` will be NULL-terminated by this function. If the buffer is too small
 * to fit the whole name, this function will return
 * `RASCAL_BUFFER_SIZE_ERROR`
 *
 * @param calculator pointer to an existing calculator
 * @param name string buffer to fill with the calculator name
 * @param bufflen number of characters available in the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_calculator_name(const struct rascal_calculator_t *calculator,
                                       char *name,
                                       uintptr_t bufflen);

/**
 * Get a copy of the parameters used to create this calculator in the
 * `parameters` buffer of size `bufflen`.
 *
 * `parameters` will be NULL-terminated by this function. If the buffer is too
 * small to fit the whole name, this function will return
 * `RASCAL_BUFFER_SIZE_ERROR`.
 *
 * @param calculator pointer to an existing calculator
 * @param parameters string buffer to fill with the parameters used to create
 *                   this calculator
 * @param bufflen number of characters available in the buffer
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_calculator_parameters(const struct rascal_calculator_t *calculator,
                                             char *parameters,
                                             uintptr_t bufflen);

/**
 * Compute the representation of the given list of `systems` with a
 * `calculator`
 *
 * This function allocates a new `eqs_tensormap_t` in `*descriptor`, which
 * memory needs to be released by the user with `eqs_tensormap_free`.
 *
 * @param calculator pointer to an existing calculator
 * @param descriptor pointer to an `eqs_tensormap_t *` that will be allocated
 *                   by this function
 * @param systems pointer to an array of systems implementation
 * @param systems_count number of systems in `systems`
 * @param options options for this calculation
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_calculator_compute(struct rascal_calculator_t *calculator,
                                          eqs_tensormap_t **descriptor,
                                          struct rascal_system_t *systems,
                                          uintptr_t systems_count,
                                          struct rascal_calculation_options_t options);

/**
 * Clear all collected profiling data
 *
 * See also `rascal_profiling_enable` and `rascal_profiling_get`.
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_profiling_clear(void);

/**
 * Enable or disable profiling data collection. By default, data collection
 * is disabled.
 *
 * Rascaline uses the [`time_graph`](https://docs.rs/time-graph/) to collect
 * timing information on the calculations. This profiling code collects the
 * total time spent inside the most important functions, as well as the
 * function call graph (which function called which other function).
 *
 * You can use `rascal_profiling_clear` to reset profiling data to an empty
 * state, and `rascal_profiling_get` to extract the profiling data.
 *
 * @param enabled whether data collection should be enabled or not
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_profiling_enable(bool enabled);

/**
 * Extract the current set of data collected for profiling.
 *
 * See also `rascal_profiling_enable` and `rascal_profiling_clear`.
 *
 * @param format in which format should the data be provided. `"table"`,
 *              `"short_table"` and `"json"` are currently supported
 * @param buffer pre-allocated buffer in which profiling data will be copied.
 *               If the buffer is too small, this function will return
 *               `RASCAL_BUFFER_SIZE_ERROR`
 * @param bufflen size of the `buffer`
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_profiling_get(const char *format, char *buffer, uintptr_t bufflen);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RASCALINE_H */
