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

typedef struct rascal_calculator_t rascal_calculator_t;

typedef struct rascal_descriptor_t rascal_descriptor_t;

typedef struct {
  uintptr_t first;
  uintptr_t second;
  double distance;
} Pair;

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
  void (*pairs)(const void *user_data, const Pair **pairs, uintptr_t *count);
} rascal_system_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

rascal_calculator_t *rascal_calculator(const char *name, const char *parameters);

void rascal_calculator_compute(rascal_calculator_t *calculator,
                               rascal_descriptor_t *descriptor,
                               rascal_system_t *systems,
                               uintptr_t count);

void rascal_calculator_free(rascal_calculator_t *calculator);

void rascal_calculator_name(const rascal_calculator_t *calculator, char *name, uintptr_t bufflen);

rascal_descriptor_t *rascal_descriptor(void);

void rascal_descriptor_free(rascal_descriptor_t *descriptor);

void rascal_descriptor_gradients(const rascal_descriptor_t *descriptor,
                                 const double **data,
                                 uintptr_t *environments,
                                 uintptr_t *features);

void rascal_descriptor_indexes(const rascal_descriptor_t *descriptor,
                               rascal_indexes indexes,
                               const uintptr_t **values,
                               uintptr_t *size,
                               uintptr_t *count);

void rascal_descriptor_indexes_names(const rascal_descriptor_t *descriptor,
                                     rascal_indexes indexes,
                                     const char **names,
                                     uintptr_t size);

void rascal_descriptor_values(const rascal_descriptor_t *descriptor,
                              const double **data,
                              uintptr_t *environments,
                              uintptr_t *features);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RASCALINE_H */
