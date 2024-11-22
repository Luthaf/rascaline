#ifndef FEATOMIC_EXAMPLE_SYSTEMS_H
#define FEATOMIC_EXAMPLE_SYSTEMS_H

#include <featomic.h>

int read_systems_example(const char* path, featomic_system_t** systems, uintptr_t* n_systems);
void free_systems_example(featomic_system_t* systems, uintptr_t n_systems);

#endif
