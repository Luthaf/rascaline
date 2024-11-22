#include <cstring>

#include "helpers.hpp"

#define SQRT_3 1.73205080756887729352

featomic_system_t simple_system() {
    featomic_system_t system = {};

    system.size = [](const void* _, uintptr_t* size) {
        *size = 4;
        return FEATOMIC_SUCCESS;
    };

    system.positions = [](const void* _, const double** positions) {
        static double POSITIONS[4][3] = {
            {10, 10, 10},
            {1, 1, 1},
            {2, 2, 2},
            {3, 3, 3},
        };
        *positions = POSITIONS[0];
        return FEATOMIC_SUCCESS;
    };

    system.types = [](const void* _, const int32_t** types) {
        static int32_t TYPES[4] = {6, 1, 1, 1};
        *types = TYPES;
        return FEATOMIC_SUCCESS;
    };

    system.cell = [](const void* _, double* cell) {
        static double CELL[3][3] = {
            {10, 0, 0},
            {0, 10, 0},
            {0, 0, 10},
        };
        std::memcpy(cell, CELL, sizeof(CELL));
        return FEATOMIC_SUCCESS;
    };

    // basic compute_neighbors, always returning the same pairs
    system.compute_neighbors = [](void* _, double cutoff) {
        if (cutoff < SQRT_3 || cutoff > 3.46410161513775458704) {
            return -1;
        } else {
            return FEATOMIC_SUCCESS;
        }
    };

    system.pairs = [](const void* _, const featomic_pair_t** pairs, uintptr_t* count) {
        static featomic_pair_t PAIRS[] = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        *pairs = PAIRS;
        *count = 3;
        return FEATOMIC_SUCCESS;
    };

    system.pairs_containing = [](const void* _, uintptr_t atom, const featomic_pair_t** pairs, uintptr_t* count){
        static featomic_pair_t PAIRS_0[] = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
        };

        static featomic_pair_t PAIRS_1[] = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        static featomic_pair_t PAIRS_2[] = {
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        static featomic_pair_t PAIRS_3[] = {
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        if (atom == 0) {
            *pairs = PAIRS_0;
            *count = 1;
        } else if (atom == 1) {
            *pairs = PAIRS_1;
            *count = 2;
        } else if (atom == 2) {
            *pairs = PAIRS_2;
            *count = 2;
        } else if (atom == 3) {
            *pairs = PAIRS_3;
            *count = 1;
        } else {
            return -1;
        }
        return FEATOMIC_SUCCESS;
    };

    return system;
}

mts_array_t empty_array(std::vector<size_t> array_shape) {
    mts_array_t array = {};

    array.ptr = new std::vector<size_t>(std::move(array_shape));
    array.origin = [](const void *array, mts_data_origin_t *origin){
        mts_register_data_origin("c-tests-empty-array", origin);
        return MTS_SUCCESS;
    };
    array.shape = [](const void *array, const uintptr_t** shape, uintptr_t* shape_count){
        const auto* array_shape = static_cast<const std::vector<size_t>*>(array);
        *shape = array_shape->data();
        *shape_count = array_shape->size();
        return MTS_SUCCESS;
    };
    array.destroy = [](void *array){
        auto* array_shape = static_cast<std::vector<size_t>*>(array);
        delete array_shape;
    };
    array.copy = [](const void *array, mts_array_t* new_array){
        const auto* array_shape = static_cast<const std::vector<size_t>*>(array);
        *new_array = empty_array(*array_shape);
        return MTS_SUCCESS;
    };

    return array;
}
