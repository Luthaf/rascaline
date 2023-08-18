#ifndef RASCAL_CXX_TEST_SYSTEMS_H
#define RASCAL_CXX_TEST_SYSTEMS_H

#include "rascaline.hpp"

#define SQRT_3 1.73205080756887729352

class TestSystem: public rascaline::System {
    uintptr_t size() const override {
        return 4;
    }

    const int32_t* species() const override {
        static int32_t SPECIES[4] = {6, 1, 1, 1};
        return &SPECIES[0];
    }

    const double* positions() const override {
        static double POSITIONS[4][3] = {
            {10, 10, 10},
            {1, 1, 1},
            {2, 2, 2},
            {3, 3, 3},
        };
        return &POSITIONS[0][0];
    }

    CellMatrix cell() const override {
        return {{
            {{10, 0, 0}},
            {{0, 10, 0}},
            {{0, 0, 10}},
        }};
    }

    // basic compute_neighbors, always returning the same pairs
    void compute_neighbors(double cutoff) override {
        assert(cutoff > SQRT_3 && cutoff < 3.46410161513775458704);
    }

    const std::vector<rascal_pair_t>& pairs() const override {
        static std::vector<rascal_pair_t> PAIRS = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };
        return PAIRS;
    }

    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override {
        static std::vector<rascal_pair_t> PAIRS_0 = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
        };

        static std::vector<rascal_pair_t> PAIRS_1 = {
            {0, 1, SQRT_3, {1.0, 1.0, 1.0}, {1, 1, 1}},
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        static std::vector<rascal_pair_t> PAIRS_2 = {
            {1, 2, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        static std::vector<rascal_pair_t> PAIRS_3 = {
            {2, 3, SQRT_3, {1.0, 1.0, 1.0}, {0, 0, 0}},
        };

        if (center == 0) {
            return PAIRS_0;
        } else if (center == 1) {
            return PAIRS_1;
        } else if (center == 2) {
            return PAIRS_2;
        } else if (center == 3) {
            return PAIRS_3;
        } else {
            throw std::runtime_error("center should be below 3");
        }
    }
};

#undef SQRT_3

#endif
