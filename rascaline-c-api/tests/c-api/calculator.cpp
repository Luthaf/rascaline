#include <vector>
#include <string>

#include "rascaline.h"
#include "catch.hpp"
#include "helpers.hpp"

static void check_indexes(
    rascal_descriptor_t* descriptor,
    rascal_indexes kind,
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
        CHECK(status == RASCAL_INVALID_PARAMETER_ERROR);

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
        CHECK(status == RASCAL_INVALID_PARAMETER_ERROR);

        CHECK_SUCCESS(rascal_calculator_parameters(calculator, buffer, 4096));
        CHECK(buffer == HYPERS_JSON);

        delete[] buffer;

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

        auto options = rascal_calculation_options_t {
            /* use_native_system */ false,
            /* selected_samples */ nullptr,
            /* selected_samples_count */ 0,
            /* selected_features */ nullptr,
            /* selected_features_count */ 0,
        };
        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        auto expected = std::vector<int32_t>{
            0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        check_indexes(descriptor, RASCAL_INDEXES_ENVIRONMENTS, {"structure", "center"}, expected, 4, 2);

        expected = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, expected, 2, 2);

        const double* data = nullptr;
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

        auto options = rascal_calculation_options_t {
            /* use_native_system */ false,
            /* selected_samples */ samples.data(),
            /* selected_samples_count */ samples.size(),
            /* selected_features */ nullptr,
            /* selected_features_count */ 0,
        };
        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        check_indexes(descriptor, RASCAL_INDEXES_ENVIRONMENTS, {"structure", "center"}, samples, 2, 2);

        auto expected = std::vector<int32_t>{
            1, 0, /**/ 0, 1
        };
        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, expected, 2, 2);

        const double* data = nullptr;
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
        auto options = rascal_calculation_options_t {
            /* use_native_system */ false,
            /* selected_samples */ nullptr,
            /* selected_samples_count */ 0,
            /* selected_features */ features.data(),
            /* selected_features_count */ features.size(),
        };
        CHECK_SUCCESS(rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        ));

        auto expected = std::vector<int32_t>{
            0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        check_indexes(descriptor, RASCAL_INDEXES_ENVIRONMENTS, {"structure", "center"}, expected, 4, 2);

        check_indexes(descriptor, RASCAL_INDEXES_FEATURES, {"index_delta", "x_y_z"}, features, 1, 2);

        const double* data = nullptr;
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
        auto options = rascal_calculation_options_t {
            /* use_native_system */ false,
            /* selected_samples */ samples.data(),
            /* selected_samples_count */ samples.size(),
            /* selected_features */ nullptr,
            /* selected_features_count */ 0,
        };
        auto status = rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        );
        CHECK(status != RASCAL_SUCCESS);
        CHECK(std::string(rascal_last_error()) == "invalid parameter: wrong size for partial samples list, expected a multiple of 2, got 3");

        auto features = std::vector<int32_t>{0, 1, 1};
        options = rascal_calculation_options_t {
            /* use_native_system */ false,
            /* selected_samples */ nullptr,
            /* selected_samples_count */ 0,
            /* selected_features */ features.data(),
            /* selected_features_count */ features.size(),
        };
        status = rascal_calculator_compute(
            calculator, descriptor, &system, 1, options
        );
        CHECK(status != RASCAL_SUCCESS);
        CHECK(std::string(rascal_last_error()) == "invalid parameter: wrong size for partial features list, expected a multiple of 2, got 3");
    }

    rascal_calculator_free(calculator);
    rascal_descriptor_free(descriptor);
}

void check_indexes(
    rascal_descriptor_t* descriptor,
    rascal_indexes kind,
    std::vector<std::string> names,
    std::vector<int32_t> values,
    uintptr_t count,
    uintptr_t size
) {
    const int32_t* actual_values = nullptr;
    uintptr_t actual_count = 0;
    uintptr_t actual_size = 0;

    CHECK_SUCCESS(rascal_descriptor_indexes(
        descriptor, kind, &actual_values, &actual_count, &actual_size
    ));
    REQUIRE(actual_values != nullptr);

    REQUIRE(values.size() == count * size);
    CHECK(actual_count == count);
    CHECK(actual_size == size);

    for (size_t i=0; i<count; i++) {
        for (size_t j=0; j<size; j++) {
            CHECK(actual_values[i * size + j] == values[i * size + j]);
        }
    }

    const char** actual_names = static_cast<const char**>(std::malloc(actual_size * sizeof(const char*)));
    rascal_descriptor_indexes_names(descriptor, kind, actual_names, actual_size);

    for (size_t i=0; i<size; i++) {
        CHECK(actual_names[i] == names[i]);
    }
    std::free(actual_names);
}
