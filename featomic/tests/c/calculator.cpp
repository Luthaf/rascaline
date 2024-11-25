#include <vector>
#include <string>
#include <cstring>

#include "featomic.h"
#include "catch.hpp"
#include "helpers.hpp"

static void check_block(
    mts_tensormap_t* descriptor,
    size_t block_id,
    const std::vector<int32_t>& samples,
    const std::vector<int32_t>& properties,
    const std::vector<double>& values,
    const std::vector<int32_t>& gradient_samples,
    const std::vector<double>& gradients
);

TEST_CASE("calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        char buffer[256] = {};
        CHECK_SUCCESS(featomic_calculator_name(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == std::string("dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar"));

        featomic_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": ")" + name + "\"}";

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = featomic_calculator_name(calculator, buffer, 256);
        CHECK(status == FEATOMIC_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(featomic_calculator_name(calculator, buffer, 4096));
        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name;
        CHECK(buffer == expected);

        delete[] buffer;

        featomic_calculator_free(calculator);
    }
}

TEST_CASE("calculator parameters") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char buffer[256] = {};
        CHECK_SUCCESS(featomic_calculator_parameters(calculator, buffer, sizeof(buffer)));
        CHECK(buffer == HYPERS_JSON);

        featomic_calculator_free(calculator);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": ")" + name + "\"}";

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        char* buffer = new char[4096];
        auto status = featomic_calculator_parameters(calculator, buffer, 256);
        CHECK(status == FEATOMIC_BUFFER_SIZE_ERROR);

        CHECK_SUCCESS(featomic_calculator_parameters(calculator, buffer, 4096));
        CHECK(buffer == HYPERS_JSON);

        delete[] buffer;

        featomic_calculator_free(calculator);
    }

    SECTION("cutoffs") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON.c_str());
        REQUIRE(calculator != nullptr);

        const double* cutoffs = nullptr;
        uintptr_t cutoffs_count = 0;
        CHECK_SUCCESS(featomic_calculator_cutoffs(calculator, &cutoffs, &cutoffs_count));
        CHECK(cutoffs_count == 1);
        CHECK(cutoffs[0] == 3.5);

        featomic_calculator_free(calculator);
    }
}

TEST_CASE("calculator creation errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": "532",
        "delta": 25,
        "name": "bar"
    })";
    auto *calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
    CHECK(calculator == nullptr);

    CHECK(std::string(featomic_last_error()) == "json error: invalid type: string \"532\", expected f64 at line 2 column 23");
}

TEST_CASE("Compute descriptor") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": ""
    })";

    SECTION("Full compute") {
        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        mts_labels_t keys;
        std::memset(&keys, 0, sizeof(mts_labels_t));

        status = mts_tensormap_keys(descriptor, &keys);
        CHECK_SUCCESS(status);

        CHECK(keys.size == 1);
        CHECK(keys.names[0] == std::string("center_type"));
        CHECK(keys.count == 2);
        CHECK(keys.values[0] == 1);
        CHECK(keys.values[1] == 6);
        mts_labels_free(&keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        auto values = std::vector<double>{
            5, 39, /**/ 6, 18, /**/ 7, 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 1, /**/ 1, 0, 2, /**/ 1, 0, 3,
            2, 0, 2, /**/ 2, 0, 3,
        };
        auto gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        values = std::vector<double>{
            4, 33,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- samples") {
        auto selected_sample_values = std::vector<int32_t>{
            0, 1, /**/ 0, 3,
        };
        auto selected_sample_names = std::vector<const char*>{
            "system", "atom"
        };

        mts_labels_t selected_samples;
        std::memset(&selected_samples, 0, sizeof(mts_labels_t));

        selected_samples.names = selected_sample_names.data();
        selected_samples.values = selected_sample_values.data();
        selected_samples.count = 2;
        selected_samples.size = 2;

        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_samples.subset = &selected_samples;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );

        CHECK_SUCCESS(status);

        mts_labels_t keys;
        std::memset(&keys, 0, sizeof(mts_labels_t));

        status = mts_tensormap_keys(descriptor, &keys);
        CHECK_SUCCESS(status);

        CHECK(keys.size == 1);
        CHECK(keys.names[0] == std::string("center_type"));
        CHECK(keys.count == 2);
        CHECK(keys.values[0] == 1);
        CHECK(keys.values[1] == 6);
        mts_labels_free(&keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            1, 0, /**/ 0, 1,
        };
        auto values = std::vector<double>{
            5, 39, /**/ 7, 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 2, /**/ 1, 0, 3,
        };
        auto gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{};
        values = std::vector<double>{};
        gradient_samples = std::vector<int32_t>{};
        gradients = std::vector<double>{};

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- features") {
        auto selected_property_values = std::vector<int32_t>{
            0, 1,
        };
        auto selected_property_names = std::vector<const char*>{
            "index_delta", "x_y_z"
        };

        mts_labels_t selected_properties;
        std::memset(&selected_properties, 0, sizeof(mts_labels_t));

        selected_properties.names = selected_property_names.data();
        selected_properties.values = selected_property_values.data();
        selected_properties.count = 1;
        selected_properties.size = 2;

        auto system = simple_system();

        featomic_calculation_options_t options;
        std::memset(&options, 0, sizeof(featomic_calculation_options_t));

        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_properties.subset = &selected_properties;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        mts_labels_t keys = {};
        status = mts_tensormap_keys(descriptor, &keys);
        CHECK_SUCCESS(status);

        CHECK(keys.size == 1);
        CHECK(keys.names[0] == std::string("center_type"));
        CHECK(keys.count == 2);
        CHECK(keys.values[0] == 1);
        CHECK(keys.values[1] == 6);
        mts_labels_free(&keys);

        auto samples = std::vector<int32_t>{
            0, 1, /**/ 0, 2, /**/ 0, 3,
        };
        auto properties = std::vector<int32_t>{
            0, 1,
        };
        auto values = std::vector<double>{
            39, /**/ 18, /**/ 15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 2,
            1, 0, 1, /**/ 1, 0, 2, /**/ 1, 0, 3,
            2, 0, 2, /**/ 2, 0, 3,
        };
        auto gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        values = std::vector<double>{
            33,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- preselected") {
        auto sample_names = std::vector<const char*>{
            "system", "atom"
        };
        auto property_names = std::vector<const char*>{
            "index_delta", "x_y_z"
        };

        auto h_sample_values = std::vector<int32_t>{
            0, 3,
        };
        auto h_property_values = std::vector<int32_t>{
            0, 1,
        };

        mts_block_t* blocks[2] = {nullptr, nullptr};

        mts_labels_t h_samples = {};
        h_samples.size = 2;
        h_samples.names = sample_names.data();
        h_samples.count = 1;
        h_samples.values = h_sample_values.data();

        mts_labels_t h_properties = {};
        h_properties.size = 2;
        h_properties.names = property_names.data();
        h_properties.count = 1;
        h_properties.values = h_property_values.data();
        blocks[0] = mts_block(empty_array({1, 1}), h_samples, nullptr, 0, h_properties);
        REQUIRE(blocks[0] != nullptr);


        auto c_sample_values = std::vector<int32_t>{
            0, 0,
        };
        auto c_property_values = std::vector<int32_t>{
            1, 0,
        };

        mts_labels_t c_samples = {};
        c_samples.size = 2;
        c_samples.names = sample_names.data();
        c_samples.count = 1;
        c_samples.values = c_sample_values.data();

        mts_labels_t c_properties = {};
        c_properties.size = 2;
        c_properties.names = property_names.data();
        c_properties.count = 1;
        c_properties.values = c_property_values.data();
        blocks[1] = mts_block(empty_array({1, 1}), c_samples, nullptr, 0, c_properties);
        REQUIRE(blocks[1] != nullptr);

        auto keys_names = std::vector<const char*>{"center_type"};
        auto keys_values = std::vector<int32_t>{1, 6};

        mts_labels_t keys = {};
        keys.size = 1;
        keys.names = keys_names.data();
        keys.count = 2;
        keys.values = keys_values.data();

        auto* predefined = mts_tensormap(keys, blocks, 2);
        REQUIRE(predefined != nullptr);

        auto system = simple_system();
        featomic_calculation_options_t options = {};
        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_samples.predefined = predefined;
        options.selected_properties.predefined = predefined;
        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        status = mts_tensormap_keys(descriptor, &keys);
        CHECK_SUCCESS(status);

        CHECK(keys.size == 1);
        CHECK(keys.names[0] == std::string("center_type"));
        CHECK(keys.count == 2);
        CHECK(keys.values[0] == 1);
        CHECK(keys.values[1] == 6);
        mts_labels_free(&keys);

        auto samples = std::vector<int32_t>{
            0, 3,
        };
        auto properties = std::vector<int32_t>{
            0, 1,
        };
        auto values = std::vector<double>{
            15,
        };
        auto gradient_samples = std::vector<int32_t>{
            0, 0, 2, /**/ 0, 0, 3,
        };
        auto gradients = std::vector<double>{
            1.0, /**/ 1.0, /**/ 1.0,
            1.0, /**/ 1.0, /**/ 1.0
        };

        // H block
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        properties = std::vector<int32_t>{
            1, 0,
        };
        values = std::vector<double>{
            4,
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, /**/ 0.0, /**/ 0.0,
            0.0, /**/ 0.0, /**/ 0.0,
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(predefined);
        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }

    SECTION("Partial compute -- key selection") {
        // check key selection: we add a non-existing key (12) and remove an
        // existing one (1) from the default set of keys. We also put the keys
        // in a different order than what would be the default (6, 12).

        mts_labels_t selected_keys = {};
        const char* sample_names[] = {"center_type"};
        selected_keys.names = sample_names;
        selected_keys.size = 1;

        int32_t sample_values[] = {12, 6};
        selected_keys.values = sample_values;
        selected_keys.count = 2;

        auto system = simple_system();

        featomic_calculation_options_t options = {};
        const char* gradients_list[] = {"positions"};
        options.gradients = gradients_list;
        options.gradients_count = 1;
        options.selected_keys = &selected_keys;

        auto* calculator = featomic_calculator("dummy_calculator", HYPERS_JSON);
        REQUIRE(calculator != nullptr);

        mts_tensormap_t* descriptor = nullptr;
        auto status = featomic_calculator_compute(
            calculator, &descriptor, &system, 1, options
        );
        CHECK_SUCCESS(status);

        mts_labels_t keys = {};
        status = mts_tensormap_keys(descriptor, &keys);
        CHECK_SUCCESS(status);

        CHECK(keys.size == 1);
        CHECK(keys.names[0] == std::string("center_type"));
        CHECK(keys.count == 2);
        CHECK(keys.values[0] == 12);
        CHECK(keys.values[1] == 6);
        mts_labels_free(&keys);

        auto samples = std::vector<int32_t>{};
        auto properties = std::vector<int32_t>{
            1, 0, 0, 1
        };
        auto values = std::vector<double>{};
        auto gradient_samples = std::vector<int32_t>{};
        auto gradients = std::vector<double>{};

        // empty block, center_type=12 is not present in the system
        check_block(descriptor, 0, samples, properties, values, gradient_samples, gradients);

        samples = std::vector<int32_t>{
            0, 0,
        };
        properties = std::vector<int32_t>{
            1, 0, 0, 1
        };
        values = std::vector<double>{
            4, 33
        };
        gradient_samples = std::vector<int32_t>{
            0, 0, 0, /**/ 0, 0, 1,
        };
        gradients = std::vector<double>{
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0
        };

        // C block
        check_block(descriptor, 1, samples, properties, values, gradient_samples, gradients);

        mts_tensormap_free(descriptor);
        featomic_calculator_free(calculator);
    }
}

void check_block(
    mts_tensormap_t* descriptor,
    size_t block_id,
    const std::vector<int32_t>& samples,
    const std::vector<int32_t>& properties,
    const std::vector<double>& values,
    const std::vector<int32_t>& gradient_samples,
    const std::vector<double>& gradients
) {
    mts_block_t* block = nullptr;

    auto status = mts_tensormap_block_by_id(descriptor, &block, block_id);
    CHECK_SUCCESS(status);

    /**************************************************************************/
    mts_labels_t labels = {};
    status = mts_block_labels(block, 0, &labels);
    CHECK_SUCCESS(status);

    CHECK(labels.size == 2);
    CHECK(labels.names[0] == std::string("system"));
    CHECK(labels.names[1] == std::string("atom"));
    auto n_samples = labels.count;

    auto label_values = std::vector<int32_t>(
        labels.values, labels.values + labels.count * labels.size
    );
    CHECK(label_values == samples);
    mts_labels_free(&labels);

    /**************************************************************************/
    status = mts_block_labels(block, 1, &labels);
    CHECK_SUCCESS(status);

    CHECK(labels.size == 2);
    CHECK(labels.names[0] == std::string("index_delta"));
    CHECK(labels.names[1] == std::string("x_y_z"));
    auto n_properties = labels.count;

    label_values = std::vector<int32_t>(
        labels.values, labels.values + labels.count * labels.size
    );
    CHECK(label_values == properties);
    mts_labels_free(&labels);

    /**************************************************************************/
    mts_array_t array = {};
    status = mts_block_data(block, &array);
    CHECK_SUCCESS(status);


    const uintptr_t* shape = nullptr;
    uintptr_t shape_count = 0;
    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);

    double* values_ptr = nullptr;
    status = array.data(array.ptr, &values_ptr);
    CHECK_SUCCESS(status);

    CHECK(shape_count == 2);
    CHECK(shape[0] == n_samples);
    CHECK(shape[1] == n_properties);

    auto actual_values = std::vector<double>(
        values_ptr, values_ptr + n_samples * n_properties
    );
    CHECK(actual_values == values);

    /**************************************************************************/
    mts_block_t* gradients_block = nullptr;
    status = mts_block_gradient(block, "positions", &gradients_block);
    CHECK_SUCCESS(status);

    status = mts_block_labels(gradients_block, 0, &labels);
    CHECK_SUCCESS(status);

    CHECK(labels.size == 3);
    CHECK(labels.names[0] == std::string("sample"));
    CHECK(labels.names[1] == std::string("system"));
    CHECK(labels.names[2] == std::string("atom"));
    auto n_gradient_samples = labels.count;

    label_values = std::vector<int32_t>(
        labels.values, labels.values + labels.count * labels.size
    );
    CHECK(label_values == gradient_samples);
    mts_labels_free(&labels);

    /**************************************************************************/
    status = mts_block_data(gradients_block, &array);
    CHECK_SUCCESS(status);

    status = array.shape(array.ptr, &shape, &shape_count);
    CHECK_SUCCESS(status);
    status = array.data(array.ptr, &values_ptr);
    CHECK_SUCCESS(status);

    CHECK(shape_count == 3);
    CHECK(shape[0] == n_gradient_samples);
    CHECK(shape[1] == 3);
    CHECK(shape[2] == n_properties);

    auto actual_gradients = std::vector<double>(
        values_ptr, values_ptr + n_gradient_samples * 3 * n_properties
    );
    CHECK(actual_gradients == gradients);
}
