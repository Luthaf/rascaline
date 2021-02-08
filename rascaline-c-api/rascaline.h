#ifndef RASCALINE_H
#define RASCALINE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
  RASCAL_INDEXES_FEATURES = 0,
  RASCAL_INDEXES_ENVIRONMENTS = 1,
  RASCAL_INDEXES_GRADIENTS = 2,
} rascal_indexes;

/*
 Status type returned by all functions in the C API.
 */
typedef enum {
  /*
   The function succeeded
   */
  RASCAL_SUCCESS = 0,
  /*
   A function got an invalid parameter
   */
  RASCAL_INVALID_PARAMETER_ERROR = 1,
  /*
   There was an error reading or writing JSON
   */
  RASCAL_JSON_ERROR = 2,
  /*
   A string contains non-utf8 data
   */
  RASCAL_UTF8_ERROR = 3,
  /*
   There was an error of unknown kind
   */
  RASCAL_UNKNOWN_ERROR = 254,
  /*
   There was an internal error (rust panic)
   */
  RASCAL_INTERNAL_PANIC = 255,
} rascal_status_t;

/*
 Opaque type representing a Calculator
 */
typedef struct rascal_calculator_t rascal_calculator_t;

/*
 Opaque type representing a Descriptor
 */
typedef struct rascal_descriptor_t rascal_descriptor_t;

/*
 Pair of atoms coming from a neighbor list
 */
typedef struct {
  /*
   index of the first atom in the pair
   */
  uintptr_t first;
  /*
   index of the second atom in the pair
   */
  uintptr_t second;
  /*
   vector from the first atom to the second atom, wrapped inside the unit
   cell as required
   */
  double vector[3];
} rascal_pair_t;

typedef struct {
  /*
   User-provided data should be stored here, it will be passed as the
   first parameter to all function pointers
   */
  void *user_data;
  void (*size)(const void *user_data, uintptr_t *size);
  void (*species)(const void *user_data, const uintptr_t **species);
  void (*positions)(const void *user_data, const double **positions);
  void (*cell)(const void *user_data, double *cell);
  void (*compute_neighbors)(void *user_data, double cutoff);
  void (*pairs)(const void *user_data, const rascal_pair_t **pairs, uintptr_t *count);
} rascal_system_t;

typedef struct {
  /*
   Copy the data from systems into native `SimpleSystem`. This can be
   faster than having to cross the FFI boundary too often.
   */
  bool use_native_system;
  /*
   List of samples on which to run the calculation. Use `NULL` to run the
   calculation on all samples.
   */
  const double *selected_samples;
  /*
   If selected_samples is not `NULL`, this should be set to the size of the
   selected_samples array
   */
  uintptr_t selected_samples_count;
  /*
   List of features on which to run the calculation. Use `NULL` to run the
   calculation on all features.
   */
  const double *selected_features;
  /*
   If selected_features is not `NULL`, this should be set to the size of the
   selected_features array
   */
  uintptr_t selected_features_count;
} rascal_calculation_options_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 Get the last error message that was sent on the current thread
 */
const char *rascal_last_error(void);

rascal_descriptor_t *rascal_descriptor(void);

rascal_status_t rascal_descriptor_free(rascal_descriptor_t *descriptor);

rascal_status_t rascal_descriptor_values(const rascal_descriptor_t *descriptor,
                                         const double **data,
                                         uintptr_t *environments,
                                         uintptr_t *features);

rascal_status_t rascal_descriptor_gradients(const rascal_descriptor_t *descriptor,
                                            const double **data,
                                            uintptr_t *environments,
                                            uintptr_t *features);

rascal_status_t rascal_descriptor_indexes(const rascal_descriptor_t *descriptor,
                                          rascal_indexes indexes,
                                          const double **values,
                                          uintptr_t *count,
                                          uintptr_t *size);

rascal_status_t rascal_descriptor_indexes_names(const rascal_descriptor_t *descriptor,
                                                rascal_indexes indexes,
                                                const char **names,
                                                uintptr_t size);

rascal_status_t rascal_descriptor_densify(rascal_descriptor_t *descriptor, const char *variable);

rascal_calculator_t *rascal_calculator(const char *name, const char *parameters);

rascal_status_t rascal_calculator_free(rascal_calculator_t *calculator);

rascal_status_t rascal_calculator_name(const rascal_calculator_t *calculator,
                                       char *name,
                                       uintptr_t bufflen);

rascal_status_t rascal_calculator_parameters(const rascal_calculator_t *calculator,
                                             char *parameters,
                                             uintptr_t bufflen);

rascal_status_t rascal_calculator_compute(rascal_calculator_t *calculator,
                                          rascal_descriptor_t *descriptor,
                                          rascal_system_t *systems,
                                          uintptr_t systems_count,
                                          rascal_calculation_options_t options);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RASCALINE_H */
