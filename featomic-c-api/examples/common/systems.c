#include <stdio.h>

#include <chemfiles.h>

#include "systems.h"


// This file shows an example on how to define a custom `featomic_system_t`,
// based on chemfiles (https://chemfiles.org).
//
// The core idea is to define a set of functions to communicate data (atomic
// types, positions, periodic cell, …) between your code and featomic. Each
// function takes a `void* user_data` parameter, which you can set to anything
// relevant.

typedef struct {
    uint64_t size;
    CHFL_FRAME* frame;
    int32_t* atom_types;
} chemfiles_system_t;

static featomic_status_t chemfiles_system_size(const void* system, uintptr_t* size) {
    *size = ((const chemfiles_system_t*)system)->size;
    return FEATOMIC_SUCCESS;
}

static featomic_status_t chemfiles_system_cell(const void* system, double cell[9]) {
    CHFL_CELL* chemfiles_cell = chfl_cell_from_frame(((const chemfiles_system_t*)system)->frame);
    if (chemfiles_cell == NULL) {
        printf("Error: %s", chfl_last_error());
        return -1;
    }

    chfl_status status = chfl_cell_matrix(chemfiles_cell, (chfl_vector3d*)cell);
    chfl_free(chemfiles_cell);

    if (status == CHFL_SUCCESS) {
        return FEATOMIC_SUCCESS;
    } else {
        printf("Error: %s", chfl_last_error());
        return -2;
    }
}

static featomic_status_t chemfiles_system_positions(const void* system, const double** positions) {
    chfl_vector3d* chemfiles_positions = NULL;
    uint64_t size = 0;
    chfl_status status = chfl_frame_positions(
        ((const chemfiles_system_t*)system)->frame,
        &chemfiles_positions,
        &size
    );

    if (status == CHFL_SUCCESS) {
        *positions = (const double*)chemfiles_positions;
        return FEATOMIC_SUCCESS;
    } else {
        return -1;
    }
}

static featomic_status_t chemfiles_system_types(const void* system, const int32_t** types) {
    *types = ((const chemfiles_system_t*)system)->atom_types;
    return FEATOMIC_SUCCESS;
}

static featomic_status_t chemfiles_system_neighbors(void* system, double cutoff) {
    // this system does not have a neighbor list, and needs to use the one
    // inside featomic with `use_native_system=true`
    return -1;
}


int read_systems_example(const char* path, featomic_system_t** systems, uintptr_t* n_systems) {
    CHFL_TRAJECTORY* trajectory = NULL;
    CHFL_ATOM* atom = NULL;
    chemfiles_system_t* system = NULL;
    chfl_status status;
    uint64_t step = 0;
    uint64_t n_steps = 0;

    trajectory = chfl_trajectory_open(path, 'r');
    if (trajectory == NULL) {
        printf("Error: %s", chfl_last_error());
        goto error;
    }

    status = chfl_trajectory_nsteps(trajectory, &n_steps);
    if (status != CHFL_SUCCESS) {
        printf("Error: %s", chfl_last_error());
        goto error;
    }

    *systems = calloc(n_steps, sizeof(featomic_system_t));
    if (*systems == NULL) {
        printf("Error: Failed to allocate systems");
        goto error;
    }
    *n_systems = (uintptr_t)n_steps;

    for (step=0; step<n_steps; step++) {
        system = calloc(1, sizeof(chemfiles_system_t));
        if (system == NULL) {
            printf("Error: failed to allocate single system");
            goto error;
        }

        system->frame = chfl_frame();
        if (system->frame == NULL) {
            printf("Error: %s", chfl_last_error());
            goto error;
        }

        status = chfl_trajectory_read_step(trajectory, step, system->frame);
        if (status != CHFL_SUCCESS) {
            printf("Error: %s", chfl_last_error());
            goto error;
        }

        // extract atomic types from the frame
        chfl_frame_atoms_count(system->frame, &system->size);
        if (status != CHFL_SUCCESS) {
            printf("Error: %s", chfl_last_error());
            goto error;
        }

        system->atom_types = calloc(system->size, sizeof(int32_t));
        if (system->atom_types == NULL) {
            printf("Error: failed to allocate atom_types");
            goto error;
        }

        for (uint64_t i=0; i<system->size; i++) {
            atom = chfl_atom_from_frame(system->frame, i);
            if (atom == NULL) {
                printf("Error: %s", chfl_last_error());
                goto error;
            }

            uint64_t number = 0;
            chfl_atom_atomic_number(atom, &number);
            system->atom_types[i] = (int32_t)(number);

            chfl_free(atom);
        }

        // setup all the data and functions for the system
        (*systems)[step].user_data = system;

        (*systems)[step].size = chemfiles_system_size;
        (*systems)[step].cell = chemfiles_system_cell;
        (*systems)[step].positions = chemfiles_system_positions;
        (*systems)[step].types = chemfiles_system_types;
        (*systems)[step].compute_neighbors = chemfiles_system_neighbors;
        (*systems)[step].pairs = NULL;
        (*systems)[step].pairs_containing = NULL;

        system = NULL;
    }

    chfl_free(trajectory);
    return 0;

error:
    chfl_free(trajectory);
    chfl_free(atom);

    // cleanup any sucesfully allocated frames
    free_systems_example(*systems, step);

    *systems = NULL;
    *n_systems = 0;

    return 1;
}

void free_systems_example(featomic_system_t* systems, uintptr_t n_systems) {
    for (size_t i=0; i<n_systems; i++) {
        chemfiles_system_t* system = systems[i].user_data;
        chfl_free(system->frame);
        free(system->atom_types);
        free(system);
    }

    free(systems);
}
