#include <cstring>
#include <cassert>

#include <stdexcept>

#include "helpers.hpp"

#define SQRT_3 1.73205080756887729352

rascal_system_t simple_system() {
    rascal_system_t system = {0};

    system.size = [](const void* _, uintptr_t* size) {
        *size = 4;
        return RASCAL_SUCCESS;
    };

    system.positions = [](const void* _, const double** positions) {
        static double POSITIONS[4][3] = {
            {0, 0, 0},
            {1, 1, 1},
            {2, 2, 2},
            {3, 3, 3},
        };
        *positions = POSITIONS[0];
        return RASCAL_SUCCESS;
    };

    system.species = [](const void* _, const int32_t** species) {
        static int32_t SPECIES[4] = {6, 1, 1, 1};
        *species = SPECIES;
        return RASCAL_SUCCESS;
    };

    system.cell = [](const void* _, double* cell) {
        static double CELL[3][3] = {
            {10, 0, 0},
            {0, 10, 0},
            {0, 0, 10},
        };
        std::memcpy(cell, CELL, sizeof(CELL));
        return RASCAL_SUCCESS;
    };

    // basic compute_neighbors, always returning the same pairs
    system.compute_neighbors = [](void* _, double cutoff) {
        if (cutoff < SQRT_3 || cutoff > 3.46410161513775458704) {
            return -1;
        } else {
            return RASCAL_SUCCESS;
        }
    };

    system.pairs = [](const void* _, const rascal_pair_t** pairs, uintptr_t* count) {
        static rascal_pair_t PAIRS[] = {
            {0, 1, SQRT_3, {1, 1, 1}},
            {1, 2, SQRT_3, {1, 1, 1}},
            {2, 3, SQRT_3, {1, 1, 1}},
        };

        *pairs = PAIRS;
        *count = 3;
        return RASCAL_SUCCESS;
    };

    system.pairs_containing = [](const void* _, uintptr_t center, const rascal_pair_t** pairs, uintptr_t* count){
        static rascal_pair_t PAIRS_0[] = {
            {0, 1, SQRT_3, {1, 1, 1}},
        };

        static rascal_pair_t PAIRS_1[] = {
            {0, 1, SQRT_3, {1, 1, 1}},
            {1, 2, SQRT_3, {1, 1, 1}},
        };

        static rascal_pair_t PAIRS_2[] = {
            {1, 2, SQRT_3, {1, 1, 1}},
            {2, 3, SQRT_3, {1, 1, 1}},
        };

        static rascal_pair_t PAIRS_3[] = {
            {2, 3, SQRT_3, {1, 1, 1}},
        };

        if (center == 0) {
            *pairs = PAIRS_0;
            *count = 1;
        } else if (center == 1) {
            *pairs = PAIRS_1;
            *count = 2;
        } else if (center == 2) {
            *pairs = PAIRS_2;
            *count = 2;
        } else if (center == 3) {
            *pairs = PAIRS_3;
            *count = 1;
        } else {
            return -1;
        }
        return RASCAL_SUCCESS;
    };

    return system;
}

mts_array_t empty_array(std::vector<size_t> array_shape) {
    mts_array_t array = {0};

    array.ptr = new std::vector<size_t>(array_shape);
    array.origin = [](const void *array, mts_data_origin_t *origin){
        mts_register_data_origin("c-tests-empty-array", origin);
        return MTS_SUCCESS;
    };
    array.shape = [](const void *array, const uintptr_t** shape, uintptr_t* shape_count){
        auto array_shape = static_cast<const std::vector<size_t>*>(array);
        *shape = array_shape->data();
        *shape_count = array_shape->size();
        return MTS_SUCCESS;
    };
    array.destroy = [](void *array){
        auto array_shape = static_cast<std::vector<size_t>*>(array);
        delete array_shape;
    };
    array.copy = [](const void *array, mts_array_t* new_array){
        auto array_shape = static_cast<const std::vector<size_t>*>(array);
        *new_array = empty_array(*array_shape);
        return MTS_SUCCESS;
    };

    return array;
}
