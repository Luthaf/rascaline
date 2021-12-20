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
 * The different kinds of indexes that can exist on a `rascal_descriptor_t`
 */
typedef enum rascal_indexes_kind {
  /**
   * The feature index, describing the features of the representation
   */
  RASCAL_INDEXES_FEATURES = 0,
  /**
   * The samples index, describing different samples in the representation
   */
  RASCAL_INDEXES_SAMPLES = 1,
  /**
   * The gradient samples index, describing the gradients of samples in the
   * representation with respect to other atoms
   */
  RASCAL_INDEXES_GRADIENT_SAMPLES = 2,
} rascal_indexes_kind;

/**
 * Opaque type representing a `Calculator`
 */
typedef struct rascal_calculator_t rascal_calculator_t;

/**
 * Opaque type representing a `Descriptor`.
 */
typedef struct rascal_descriptor_t rascal_descriptor_t;

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
 * Indexes representing metadata associated with either samples or features in
 * a given descriptor.
 */
typedef struct rascal_indexes_t {
  /**
   * Names of the variables composing this set of indexes. There are `size`
   * elements in this array, each being a NULL terminated string.
   */
  const char *const *names;
  /**
   * Pointer to the first element of a 2D row-major array of 32-bit signed
   * integer containing the values taken by the different variables in
   * `names`. Each row has `size` elements, and there are `count` rows in
   * total.
   */
  const int32_t *values;
  /**
   * Number of variables/size of a single entry in the set of indexes
   */
  uintptr_t size;
  /**
   * Number entries in the set of indexes
   */
  uintptr_t count;
} rascal_indexes_t;

/**
 * `rascal_densified_position_t` contains all the information to reconstruct
 * the new position of the values associated with a single sample in the
 * initial descriptor after a call to `rascal_descriptor_densify_values`
 */
typedef struct rascal_densified_position_t {
  /**
   * if `used` is `true`, index of the new sample in the value array
   */
  uintptr_t new_sample;
  /**
   * if `used` is `true`, index of the feature block in the new array
   */
  uintptr_t feature_block;
  /**
   * indicate whether this sample was needed to construct the new value
   * array. This might be `false` when the value of densified variables
   * specified by the user does not match the sample.
   */
  bool used;
} rascal_densified_position_t;

/**
 * Options that can be set to change how a calculator operates.
 */
typedef struct rascal_calculation_options_t {
  /**
   * Copy the data from systems into native `SimpleSystem`. This can be
   * faster than having to cross the FFI boundary too often.
   */
  bool use_native_system;
  /**
   * List of samples on which to run the calculation. You can set
   * `selected_samples.values` to `NULL` to run the calculation on all
   * samples. If necessary, gradients samples will be derived from the
   * values given in selected_samples.
   */
  struct rascal_indexes_t selected_samples;
  /**
   * List of features on which to run the calculation. You can set
   * `selected_features.values` to `NULL` to run the calculation on all
   * features.
   */
  struct rascal_indexes_t selected_features;
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
 * Create a new empty descriptor.
 *
 * All memory allocated by this function can be released using
 * `rascal_descriptor_free`.
 *
 * @returns A pointer to the newly allocated descriptor, or a `NULL` pointer in
 *          case of error. In case of error, you can use `rascal_last_error()`
 *          to get the error message.
 */
struct rascal_descriptor_t *rascal_descriptor(void);

/**
 * Free the memory associated with a `descriptor` previously created with
 * `rascal_descriptor`.
 *
 * If `descriptor` is `NULL`, this function does nothing.
 *
 * @param descriptor pointer to an existing descriptor, or `NULL`
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the
 *          full error message.
 */
rascal_status_t rascal_descriptor_free(struct rascal_descriptor_t *descriptor);

/**
 * Get the values stored inside this descriptor after a call to
 * `rascal_calculator_compute`.
 *
 * This function sets `*data` to a pointer containing the address of first
 * element of the 2D array containing the values, `*samples` to the size of the
 * first axis of this array and `*features` to the size of the second axis of
 * the array. The array is stored using a row-major layout.
 *
 * @param descriptor pointer to an existing descriptor
 * @param data pointer to a pointer to a double, will be set to the address of
 *             the first element in the values array
 * @param samples pointer to a single integer, will be set to the first
 *                dimension of the values array
 * @param features pointer to a single integer, will be set to the second
 *                 dimension of the values array
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_descriptor_values(struct rascal_descriptor_t *descriptor,
                                         double **data,
                                         uintptr_t *samples,
                                         uintptr_t *features);

/**
 * Get the gradients stored inside this descriptor after a call to
 * `rascal_calculator_compute`, if any.
 *
 * This function sets `*data` to to a pointer containing the address of the
 * first element of the 2D array containing the gradients, `*gradient_samples`
 * to the size of the first axis of this array and `*features` to the size of
 * the second axis of the array. The array is stored using a row-major layout.
 *
 * If this descriptor does not contain gradient data, `*data` is set to `NULL`,
 * while `*gradient_samples` and `*features` are set to 0.
 *
 * @param descriptor pointer to an existing descriptor
 * @param data pointer to a pointer to a double, will be set to the address of
 *             the first element in the gradients array
 * @param gradient_samples pointer to a single integer, will be set to the first
 *                         dimension of the gradients array
 * @param features pointer to a single integer, will be set to the second
 *                 dimension of the gradients array
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_descriptor_gradients(struct rascal_descriptor_t *descriptor,
                                            double **data,
                                            uintptr_t *gradient_samples,
                                            uintptr_t *features);

/**
 * Get the values associated with one of the `indexes` in the given
 * `descriptor`.
 *
 * This function sets `indexes->names` to to a **read only** array containing
 * the names of the variables in this set of indexes; `indexes->values` to to a
 * **read only** 2D array containing values taken by these variables,
 * `indexes->count` to the number of indexes (first dimension of the array) and
 * `indexes->values` to the size of each index (second dimension of the array).
 * The array is stored using a row-major layout.
 *
 * If this `descriptor` does not contain gradient data, and `indexes` is
 * `RASCAL_INDEXES_GRADIENTS`, all members of `indexes` are set to `NULL` or 0.
 *
 * @param descriptor pointer to an existing descriptor
 * @param kind type of indexes requested
 * @param indexes pointer to `rascal_indexes_t` that will be filled by this function
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_descriptor_indexes(const struct rascal_descriptor_t *descriptor,
                                          enum rascal_indexes_kind kind,
                                          struct rascal_indexes_t *indexes);

/**
 * Make the given `descriptor` dense along the given `variables`.
 *
 * The `variable` array should contain the name of the variables as
 * NULL-terminated strings, and `variables_count` must be the number of
 * variables in the array.
 *
 * The `requested` parameter defines which set of values taken by the
 * `variables` should be part of the new features. If it is `NULL`, this is the
 * set of values taken by the variables in the samples. Otherwise, it must be a
 * pointer to the first element of a 2D row-major array with one row for each
 * new feature block, and one column for each variable. `requested_size` must
 * be the number of rows in this array.
 *
 * This function "moves" the variables from the samples to the features,
 * filling the new features with zeros if the corresponding sample is missing.
 *
 * For example, take a descriptor containing two samples variables (`structure`
 * and `species`) and two features (`n` and `l`). Starting with this
 * descriptor:
 *
 * ```text
 *                       +---+---+---+
 *                       | n | 0 | 1 |
 *                       +---+---+---+
 *                       | l | 0 | 1 |
 * +-----------+---------+===+===+===+
 * | structure | species |           |
 * +===========+=========+   +---+---+
 * |     0     |    1    |   | 1 | 2 |
 * +-----------+---------+   +---+---+
 * |     0     |    6    |   | 3 | 4 |
 * +-----------+---------+   +---+---+
 * |     1     |    6    |   | 5 | 6 |
 * +-----------+---------+   +---+---+
 * |     1     |    8    |   | 7 | 8 |
 * +-----------+---------+---+---+---+
 * ```
 *
 * Calling `descriptor.densify(["species"])` will move `species` out of the
 * samples and into the features, producing:
 * ```text
 *             +---------+-------+-------+-------+
 *             | species |   1   |   6   |   8   |
 *             +---------+---+---+---+---+---+---+
 *             |    n    | 0 | 1 | 0 | 1 | 0 | 1 |
 *             +---------+---+---+---+---+---+---+
 *             |    l    | 0 | 1 | 0 | 1 | 0 | 1 |
 * +-----------+=========+===+===+===+===+===+===+
 * | structure |
 * +===========+         +---+---+---+---+---+---+
 * |     0     |         | 1 | 2 | 3 | 4 | 0 | 0 |
 * +-----------+         +---+---+---+---+---+---+
 * |     1     |         | 0 | 0 | 5 | 6 | 7 | 8 |
 * +-----------+---------+---+---+---+---+---+---+
 * ```
 *
 * Notice how there is only one row/sample for each structure now, and how each
 * value for `species` have created a full block of features. Missing values
 * (e.g. structure 0/species 8) have been filled with 0.
 */
rascal_status_t rascal_descriptor_densify(struct rascal_descriptor_t *descriptor,
                                          const char *const *variables,
                                          uintptr_t variables_count,
                                          const int32_t *requested,
                                          uintptr_t requested_size);

/**
 * Make this descriptor dense along the given `variables`, only modifying the
 * values array, and not the gradients array.
 *
 * This function behaves similarly to `rascal_descriptor_densify`, please refer
 * to its documentation for more information.
 *
 * If this descriptor contains gradients, `gradients_positions` will point to
 * an array allocated with `malloc` containing the changes made to the values
 * array, which can be used to reconstruct the change to make to the gradients.
 * Users of this function are expected to `free` the corresponding memory when
 * they no longer need it.
 *
 * This is an advanced function most users should not need to use, used to
 * implement backward propagation without having to densify the full gradient
 * array.
 */
rascal_status_t rascal_descriptor_densify_values(struct rascal_descriptor_t *descriptor,
                                                 const char *const *variables,
                                                 uintptr_t variables_count,
                                                 const int32_t *requested,
                                                 uintptr_t requested_size,
                                                 struct rascal_densified_position_t **densified_positions,
                                                 uintptr_t *densified_positions_count);

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
 * Get the default number of features this `calculator` will produce in the
 * `features` parameter.
 *
 * This number corresponds to the size of second dimension of the `values` and
 * `gradients` arrays in the `rascal_descriptor_t` after a call to
 * `rascal_calculator_compute`.
 *
 * @param calculator pointer to an existing calculator
 * @param features pointer to an integer to be filled with the number of features
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_calculator_features_count(const struct rascal_calculator_t *calculator,
                                                 uintptr_t *features);

/**
 * Run a calculation with the given `calculator` on the given `systems`,
 * storing the resulting data in the `descriptor`.
 *
 * @param calculator pointer to an existing calculator
 * @param descriptor pointer to an existing descriptor for data storage
 * @param systems pointer to an array of systems implementation
 * @param systems_count number of systems in `systems`
 * @param options options for this calculation
 *
 * @returns The status code of this operation. If the status is not
 *          `RASCAL_SUCCESS`, you can use `rascal_last_error()` to get the full
 *          error message.
 */
rascal_status_t rascal_calculator_compute(struct rascal_calculator_t *calculator,
                                          struct rascal_descriptor_t *descriptor,
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
