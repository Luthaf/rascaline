#ifndef RASCALINE_H
#define RASCALINE_H

/* Generated with cbindgen:0.13.1 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct rascal_calculator_t rascal_calculator_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

rascal_calculator_t *rascal_calculator(const char *name, const char *parameters);

void rascal_calculator_free(rascal_calculator_t *calculator);

void rascal_calculator_name(const rascal_calculator_t *calculator, char *name, uintptr_t bufflen);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RASCALINE_H */
