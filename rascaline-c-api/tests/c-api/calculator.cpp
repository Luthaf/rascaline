#include <vector>
#include <string>
#include <cstring>

#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"

static void check_indexes(
    rascal_descriptor_t* descriptor,
    rascal_indexes_kind kind,
    std::vector<std::string> names,
    std::vector<int32_t> values,
    uintptr_t count,
    uintptr_t size
);

TEST_CASE("calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        char buffer[256] = {0};
        CHECK_SUCCESS(rascal_calculator_name(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == std::string("dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar - gradients: false"));

        rascal_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = rascal_calculator_name(calculator, buffer, 256);
        CHECK(status == RASCAL_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(rascal_calculator_name(calculator, buffer, 4096));
        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name + " - gradients: false";
        CHECK(buffer == expected);

        delete[] buffer;

        rascal_calculator_free(calculator);
    }
}

TEST_CASE("calculator parameters") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char buffer[256] = {0};
        CHECK_SUCCESS(rascal_calculator_parameters(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == HYPERS_JSON);

        rascal_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = rascal_calculator_parameters(calculator, buffer, 256);
        CHECK(status == RASCAL_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(rascal_calculator_parameters(calculator, buffer, 4096));
        CHECK(buffer == HYPERS_JSON);

        delete[] buffer;

        rascal_calculator_free(calculator);
    }
}

TEST_CASE("calculator features count") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        uintptr_t count = 0;
        CHECK_SUCCESS(rascal_calculator_features_count(calculator, &count));
        CHECK(count == 2);

        rascal_calculator_free(calculator);
    }

    SECTION("sorted distances vector") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "max_neighbors": 25
        })";
        auto* calculator = rascal_calculator("sorted_distances", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        uintptr_t count = 0;
        CHECK_SUCCESS(rascal_calculator_features_count(calculator, &count));
        CHECK(count == 25);

        rascal_calculator_free(calculator);
    }
}

TEST_CASE("calculator creation errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": "532",
        "delta": 25,
        "name": "bar",
        "gradients": false
    })";
    auto *calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    CHECK(calculator == nullptr);

    CHECK(std::string(rascal_last_error()) == "json error: invalid type: string \"532\", expected f64 at line 2 column 23");
}

TEST_CASE("Compute descriptor") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "",
        "gradients": true
    })";

    auto* descriptor = rascal_descriptor();
    REQUIRE(descriptor != nullptr);
    auto* calculator = rascal_calculator("dummy_calculator", HYPERS_JSON);
    REQUIRE(calculator != nullptr);

    SECTION("Full compute") {
        auto system = simple_system();

        rascal_calculation_options_t options = {0};
        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        auto expected = std::vector<int32_t>{
            0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        check_indexes(descriptor, RASCAL_INDEXES_SAMPLES, {"structure", "center"}, expected, 4, 2);

        expected = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, expected, 2, 2);

        double* data = nullptr;
        uintptr_t shape[2] = {0};
        CHECK_SUCCESS(rascal_descriptor_values(descriptor, &data, &shape[0], &shape[1]));

        CHECK(shape[0] == 4);
        CHECK(shape[1] == 2);
        auto expected_data = std::vector<double>{
            4, 3, /**/ 5, 9, /**/ 6, 18, /**/ 7, 15,
        };
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }

        CHECK_SUCCESS(rascal_descriptor_gradients(descriptor, &data, &shape[0], &shape[1]));
        CHECK(shape[0] == 18);
        CHECK(shape[1] == 2);
        expected_data = std::vector<double>{
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
        };
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- samples") {
        auto system = simple_system();

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 3,
        };
        auto names = std::vector<const char*>{
            "structure", "center"
        };

        rascal_calculation_options_t options = {0};
        options.selected_samples.names = names.data();
        options.selected_samples.values = samples.data();
        options.selected_samples.count = 2;
        options.selected_samples.size = 2;

        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        check_indexes(descriptor, RASCAL_INDEXES_SAMPLES, {"structure", "center"}, samples, 2, 2);

        auto expected = std::vector<int32_t>{
            1, 0, /**/ 0, 1
        };
        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, expected, 2, 2);

        double* data = nullptr;
        uintptr_t shape[2] = {0};
        CHECK_SUCCESS(rascal_descriptor_values(descriptor, &data, &shape[0], &shape[1]));

        CHECK(shape[0] == 2);
        CHECK(shape[1] == 2);

        auto expected_data = std::vector<double>{
            5, 9, /**/ 7, 15,
        };
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }

        CHECK_SUCCESS(rascal_descriptor_gradients(descriptor, &data, &shape[0], &shape[1]));
        CHECK(shape[0] == 9);
        CHECK(shape[1] == 2);
        expected_data = std::vector<double>{
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1,
        };
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- features") {
        auto system = simple_system();

        auto features = std::vector<int32_t>{
            0, 1
        };
        auto names = std::vector<const char*> {
            "index_delta", "x_y_z"
        };
        rascal_calculation_options_t options = {0};
        options.selected_features.names = names.data();
        options.selected_features.size = 2;
        options.selected_features.values = features.data();
        options.selected_features.count = 1;

        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        auto expected = std::vector<int32_t>{
            0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        check_indexes(descriptor, RASCAL_INDEXES_SAMPLES, {"structure", "center"}, expected, 4, 2);

        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, features, 1, 2);

        double* data = nullptr;
        uintptr_t shape[2] = {0};
        CHECK_SUCCESS(rascal_descriptor_values(descriptor, &data, &shape[0], &shape[1]));

        CHECK(shape[0] == 4);
        CHECK(shape[1] == 1);

        auto expected_data = std::vector<double>{
            3, /**/ 9, /**/ 18, /**/ 15,
        };
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }

        CHECK_SUCCESS(rascal_descriptor_gradients(descriptor, &data, &shape[0], &shape[1]));
        CHECK(shape[0] == 18);
        CHECK(shape[1] == 1);
        expected_data = std::vector<double>(18, 1.0);
        for (size_t i=0; i<shape[0]; i++) {
            for (size_t j=0; j<shape[1]; j++) {
                CHECK(data[i * shape[1] + j] == expected_data[i * shape[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- errors") {
        auto system = simple_system();

        auto samples = std::vector<int32_t>{0, 1, 3};
        auto names = std::vector<const char*> {
            "structure", "center", "species"
        };

        rascal_calculation_options_t options = {0};
        options.selected_samples.names = names.data();
        options.selected_samples.size = 3;
        options.selected_samples.values = samples.data();
        options.selected_samples.count = 1;

        auto status = rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        );
        CHECK(status != RASCAL_SUCCESS);
        CHECK(std::string(rascal_last_error()) == "invalid parameter: 'species' in requested samples is not part of the samples of this calculator");

        auto features = std::vector<int32_t>{0, 1, 1};
        names = std::vector<const char*> {
            "index_delta", "x_y_z", "foo"
        };
        std::memset(&options, 0, sizeof(rascal_calculation_options_t));
        options.selected_features.names = names.data();
        options.selected_features.size = 3;
        options.selected_features.values = features.data();
        options.selected_features.count = 1;

        status = rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        );
        CHECK(status != RASCAL_SUCCESS);
        CHECK(std::string(rascal_last_error()) == "invalid parameter: 'foo' in requested features is not part of the features of this calculator");
    }

    rascal_calculator_free(calculator);
    rascal_descriptor_free(descriptor);
}

void check_indexes(
    rascal_descriptor_t* descriptor,
    rascal_indexes_kind kind,
    std::vector<std::string> names,
    std::vector<int32_t> values,
    uintptr_t count,
    uintptr_t size
) {
    rascal_indexes_t actual = {0};
    CHECK_SUCCESS(rascal_descriptor_indexes(descriptor, kind, &actual));
    REQUIRE(actual.values != nullptr);

    REQUIRE(values.size() == count * size);
    CHECK(actual.count == count);
    CHECK(actual.size == size);

    for (size_t i=0; i<count; i++) {
        for (size_t j=0; j<size; j++) {
            CHECK(actual.values[i * size + j] == values[i * size + j]);
        }
    }

    for (size_t i=0; i<size; i++) {
        CHECK(std::string(actual.names[i]) == names[i]);
    }
}
