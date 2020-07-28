#ifndef RASCALINE_H
#define RASCALINE_H

/* Generated with cbindgen:0.13.1 */

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
   There was an error reading or writting JSON
   */
  RASCAL_JSON_ERROR = 2,
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

typedef struct {
  uintptr_t first;
  uintptr_t second;
  double distance;
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

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

rascal_calculator_t *rascal_calculator(const char *name, const char *parameters);

rascal_status_t rascal_calculator_compute(rascal_calculator_t *calculator,
                                          rascal_descriptor_t *descriptor,
                                          rascal_system_t *systems,
                                          uintptr_t count);

rascal_status_t rascal_calculator_free(rascal_calculator_t *calculator);

rascal_status_t rascal_calculator_name(const rascal_calculator_t *calculator,
                                       char *name,
                                       uintptr_t bufflen);

rascal_status_t rascal_calculator_parameters(const rascal_calculator_t *calculator,
                                             char *parameters,
                                             uintptr_t bufflen);

rascal_descriptor_t *rascal_descriptor(void);

rascal_status_t rascal_descriptor_free(rascal_descriptor_t *descriptor);

rascal_status_t rascal_descriptor_gradients(const rascal_descriptor_t *descriptor,
                                            const double **data,
                                            uintptr_t *environments,
                                            uintptr_t *features);

rascal_status_t rascal_descriptor_indexes(const rascal_descriptor_t *descriptor,
                                          rascal_indexes indexes,
                                          const uintptr_t **values,
                                          uintptr_t *count,
                                          uintptr_t *size);

rascal_status_t rascal_descriptor_indexes_names(const rascal_descriptor_t *descriptor,
                                                rascal_indexes indexes,
                                                const char **names,
                                                uintptr_t size);

rascal_status_t rascal_descriptor_values(const rascal_descriptor_t *descriptor,
                                         const double **data,
                                         uintptr_t *environments,
                                         uintptr_t *features);

/*
 Get the last error message that was sent on the current thread
 */
const char *rascal_last_error(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RASCALINE_H */
