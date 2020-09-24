#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"

const char* HYPERS_JSON = R"({
    "cutoff": 3.0,
    "delta": 5,
    "name": "bar",
    "gradients": true
})";

static void compute_descriptor(rascal_descriptor_t* descriptor) {
    auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator);
    auto system = simple_system();
    CHECK_SUCCESS(rascal_calculator_compute(calculator, descriptor, &system, 1));
    CHECK_SUCCESS(rascal_calculator_free(calculator));
}

TEST_CASE("rascal_descriptor_t") {
    SECTION("features") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        const uintptr_t* data = nullptr;
        uintptr_t count = 0;
        uintptr_t size = 0;

        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_FEATURES, &data, &count, &size
        ));
        CHECK(data == nullptr);
        CHECK(count == 0);
        CHECK(size == 0);

        const char* names[2] = {"foo", "bar"};
        CHECK_SUCCESS(rascal_descriptor_indexes_names(
            descriptor, RASCAL_INDEXES_FEATURES, names, 2
        ));
        CHECK(names[0] == nullptr);
        CHECK(names[1] == nullptr);

        compute_descriptor(descriptor);
        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_FEATURES, &data, &count, &size
        ));
        CHECK(data != nullptr);
        CHECK(count == 2);
        CHECK(size == 2);

        CHECK(data[0 * size + 0] == 1);
        CHECK(data[0 * size + 1] == 0);
        CHECK(data[1 * size + 0] == 0);
        CHECK(data[1 * size + 1] == 1);

        CHECK_SUCCESS(rascal_descriptor_indexes_names(
            descriptor, RASCAL_INDEXES_FEATURES, names, 2
        ));
        CHECK(names[0] == std::string("index_delta"));
        CHECK(names[1] == std::string("x_y_z"));

        rascal_descriptor_free(descriptor);
    }

    SECTION("environments") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        const uintptr_t* data = nullptr;
        uintptr_t count = 0;
        uintptr_t size = 0;

        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_ENVIRONMENTS, &data, &count, &size
        ));
        CHECK(data == nullptr);
        CHECK(count == 0);
        CHECK(size == 0);

        const char* names[2] = {"foo", "bar"};
        rascal_descriptor_indexes_names(descriptor, RASCAL_INDEXES_ENVIRONMENTS, names, 2);
        CHECK(names[0] == nullptr);
        CHECK(names[1] == nullptr);


        compute_descriptor(descriptor);
        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_ENVIRONMENTS, &data, &count, &size
        ));
        CHECK(data != nullptr);
        CHECK(count == 4);
        CHECK(size == 2);

        for (size_t i=0; i<count; i++) {
            // structure 0, atom i
            CHECK(data[i * size + 0] == 0);
            CHECK(data[i * size + 1] == i);
        }

        CHECK_SUCCESS(rascal_descriptor_indexes_names(
            descriptor, RASCAL_INDEXES_ENVIRONMENTS, names, 2
        ));
        CHECK(names[0] == std::string("structure"));
        CHECK(names[1] == std::string("center"));

        rascal_descriptor_free(descriptor);
    }

    SECTION("values") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        const double* data = nullptr;
        uintptr_t shape[2] = {0};
        CHECK_SUCCESS(rascal_descriptor_values(descriptor, &data, &shape[0], &shape[1]));
        CHECK(data == nullptr);
        CHECK(shape[0] == 0);
        CHECK(shape[1] == 0);

        compute_descriptor(descriptor);
        CHECK_SUCCESS(rascal_descriptor_values(descriptor, &data, &shape[0], &shape[1]));
        CHECK(shape[0] == 4);
        CHECK(shape[1] == 2);

        CHECK(data[0 * shape[1] + 0] == 5);
        CHECK(data[0 * shape[1] + 1] == 0);

        CHECK(data[1 * shape[1] + 0] == 6);
        CHECK(data[1 * shape[1] + 1] == 3);

        CHECK(data[2 * shape[1] + 0] == 7);
        CHECK(data[2 * shape[1] + 1] == 6);

        CHECK(data[3 * shape[1] + 0] == 8);
        CHECK(data[3 * shape[1] + 1] == 9);

        CHECK_SUCCESS(rascal_descriptor_free(descriptor));
    }

    SECTION("gradient indexes") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        const uintptr_t* data = nullptr;
        uintptr_t count = 0;
        uintptr_t size = 0;

        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_GRADIENTS, &data, &count, &size
        ));
        CHECK(data == nullptr);
        CHECK(count == 0);
        CHECK(size == 0);

        const char* names[4] = {"foo", "bar", "fizz", "buzz"};
        rascal_descriptor_indexes_names(descriptor, RASCAL_INDEXES_GRADIENTS, names, 4);
        CHECK(names[0] == nullptr);
        CHECK(names[1] == nullptr);
        CHECK(names[2] == nullptr);
        CHECK(names[3] == nullptr);


        compute_descriptor(descriptor);
        CHECK_SUCCESS(rascal_descriptor_indexes(
            descriptor, RASCAL_INDEXES_GRADIENTS, &data, &count, &size
        ));
        CHECK(data != nullptr);
        CHECK(count == 18);
        CHECK(size == 4);

        auto expected = std::vector<double> {
            // structure, atom, neighbor atom, spatial
            /* x */ 0, 0, 1, 0, /* y */ 0, 0, 1, 1, /* z */ 0, 0, 1, 2,
            /* x */ 0, 1, 0, 0, /* y */ 0, 1, 0, 1, /* z */ 0, 1, 0, 2,
            /* x */ 0, 1, 2, 0, /* y */ 0, 1, 2, 1, /* z */ 0, 1, 2, 2,
            /* x */ 0, 2, 1, 0, /* y */ 0, 2, 1, 1, /* z */ 0, 2, 1, 2,
            /* x */ 0, 2, 3, 0, /* y */ 0, 2, 3, 1, /* z */ 0, 2, 3, 2,
            /* x */ 0, 3, 2, 0, /* y */ 0, 3, 2, 1, /* z */ 0, 3, 2, 2,
        };

        CHECK(std::vector<double>(data, data + (count * size)) == expected);

        CHECK_SUCCESS(rascal_descriptor_indexes_names(
            descriptor, RASCAL_INDEXES_GRADIENTS, names, 4
        ));
        CHECK(names[0] == std::string("structure"));
        CHECK(names[1] == std::string("center"));
        CHECK(names[2] == std::string("neighbor"));
        CHECK(names[3] == std::string("spatial"));

        rascal_descriptor_free(descriptor);
    }

    SECTION("gradient values") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        const double* data = nullptr;
        uintptr_t shape[2] = {0};
        CHECK_SUCCESS(rascal_descriptor_gradients(descriptor, &data, &shape[0], &shape[1]));
        CHECK(data == nullptr);
        CHECK(shape[0] == 0);
        CHECK(shape[1] == 0);

        compute_descriptor(descriptor);
        CHECK_SUCCESS(rascal_descriptor_gradients(descriptor, &data, &shape[0], &shape[1]));
        CHECK(shape[0] == 18);
        CHECK(shape[1] == 2);

        for (size_t i=0; i<shape[0]; i++) {
            CHECK(data[i * shape[1] + 0] == 0);
            CHECK(data[i * shape[1] + 1] == 1);
        }

        CHECK_SUCCESS(rascal_descriptor_free(descriptor));
    }

    SECTION("densify") {
        auto* descriptor = rascal_descriptor();
        REQUIRE(descriptor != nullptr);

        compute_descriptor(descriptor);

        const double* data = nullptr;
        uintptr_t environments = 0;
        uintptr_t features = 0;
        CHECK_SUCCESS(rascal_descriptor_values(
            descriptor, &data, &environments, &features
        ));
        CHECK(data != nullptr);
        CHECK(environments == 4);
        CHECK(features == 2);

        CHECK_SUCCESS(rascal_descriptor_densify(descriptor, "center"));

        CHECK_SUCCESS(rascal_descriptor_values(
            descriptor, &data, &environments, &features
        ));
        CHECK(data != nullptr);
        CHECK(environments == 1);
        CHECK(features == 8);

        compute_descriptor(descriptor);
        CHECK(rascal_descriptor_densify(descriptor, "not there") != RASCAL_SUCCESS);

        rascal_descriptor_free(descriptor);
    }
}
