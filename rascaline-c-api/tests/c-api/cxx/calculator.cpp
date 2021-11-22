#include <vector>
#include <string>

#include "rascaline.hpp"
#include "catch.hpp"

#include "test_system.hpp"

static void check_indexes(
    const rascaline::Indexes& indexes,
    std::vector<std::string> names,
    std::array<size_t, 2> shape,
    std::vector<int32_t> values
);

TEST_CASE("Calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

        CHECK(calculator.name() == "dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar - gradients: false");
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name + " - gradients: false";
        CHECK(calculator.name() == expected);
    }
}

TEST_CASE("Calculator parameters") {
    SECTION("dummy_calculator") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);
        CHECK(calculator.parameters() == HYPERS_JSON);
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "gradients": false,
            "name": ")" + name + "\"}";

        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);
        CHECK(calculator.parameters() == HYPERS_JSON);
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
        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);
        CHECK(calculator.features_count() == 2);
    }

    SECTION("sorted distances vector") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "max_neighbors": 25
        })";
        auto calculator = rascaline::Calculator("sorted_distances", HYPERS_JSON);
        CHECK(calculator.features_count() == 25);
    }
}

TEST_CASE("calculator creation errors") {
    const char* HYPERS_JSON = R"({
        "cutoff": "532",
        "delta": 25,
        "name": "bar",
        "gradients": false
    })";

    CHECK_THROWS_WITH(
        rascaline::Calculator("dummy_calculator", HYPERS_JSON),
        "json error: invalid type: string \"532\", expected f64 at line 2 column 23"
    );
}

TEST_CASE("Compute descriptor") {
    const char* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "",
        "gradients": true
    })";

    auto system = TestSystem();
    auto systems = std::vector<rascaline::System*>();
    systems.push_back(&system);
    auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

    SECTION("Full compute") {
        auto descriptor = calculator.compute(systems);

        check_indexes(
            descriptor.samples(),
            {"structure", "center"},
            {4, 2},
            {0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3}
        );

        check_indexes(
            descriptor.features(),
            {"index_delta", "x_y_z"},
            {2, 2},
            {1, 0, /**/ 0, 1}
        );

        auto values = descriptor.values();
        CHECK(values.shape() == std::array<size_t, 2>{4, 2});
        auto expected_data = std::vector<double>{
            4, 3, /**/ 5, 9, /**/ 6, 18, /**/ 7, 15,
        };
        for (size_t i=0; i<values.shape()[0]; i++) {
            for (size_t j=0; j<values.shape()[1]; j++) {
                CHECK(values(i, j) == expected_data[i * values.shape()[1] + j]);
            }
        }

        auto gradients = descriptor.gradients();
        CHECK(gradients.shape() == std::array<size_t, 2>{18, 2});
        expected_data = std::vector<double>{
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1, /**/ 0, 1,
        };
        for (size_t i=0; i<gradients.shape()[0]; i++) {
            for (size_t j=0; j<gradients.shape()[1]; j++) {
                CHECK(gradients(i, j) == expected_data[i * gradients.shape()[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- samples") {
        auto options = rascaline::CalculationOptions();
        options.selected_samples = rascaline::SelectedIndexes({"structure", "center"});
        options.selected_samples.add({0, 1});
        options.selected_samples.add({0, 3});

        auto descriptor = calculator.compute(systems, std::move(options));

        check_indexes(
            descriptor.samples(),
            {"structure", "center"},
            {2, 2},
            {0, 1, /**/ 0, 3}
        );

        check_indexes(
            descriptor.features(),
            {"index_delta", "x_y_z"},
            {2, 2},
            {1, 0, /**/ 0, 1}
        );

        auto values = descriptor.values();
        CHECK(values.shape() == std::array<size_t, 2>{2, 2});
        auto expected_data = std::vector<double>{
            5, 9, /**/ 7, 15,
        };
        for (size_t i=0; i<values.shape()[0]; i++) {
            for (size_t j=0; j<values.shape()[1]; j++) {
                CHECK(values(i, j) == expected_data[i * values.shape()[1] + j]);
            }
        }

        auto gradients = descriptor.gradients();
        CHECK(gradients.shape() == std::array<size_t, 2>{9, 2});
        expected_data = std::vector<double>{
            0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1,
            0, 1, /**/ 0, 1, /**/ 0, 1,
        };
        for (size_t i=0; i<gradients.shape()[0]; i++) {
            for (size_t j=0; j<gradients.shape()[1]; j++) {
                CHECK(gradients(i, j) == expected_data[i * gradients.shape()[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- features") {
        auto options = rascaline::CalculationOptions();
        options.selected_features = rascaline::SelectedIndexes({"index_delta", "x_y_z"});
        options.selected_features.add({0, 1});

        auto descriptor = calculator.compute(systems, std::move(options));

        check_indexes(
            descriptor.samples(),
            {"structure", "center"},
            {4, 2},
            {0, 0, /**/ 0, 1, /**/ 0, 2, /**/ 0, 3}
        );

        check_indexes(
            descriptor.features(),
            {"index_delta", "x_y_z"},
            {1, 2},
            {0, 1}
        );

        auto values = descriptor.values();
        CHECK(values.shape() == std::array<size_t, 2>{4, 1});
        auto expected_data = std::vector<double>{
            3, /**/ 9, /**/ 18, /**/ 15,
        };
        for (size_t i=0; i<values.shape()[0]; i++) {
            for (size_t j=0; j<values.shape()[1]; j++) {
                CHECK(values(i, j) == expected_data[i * values.shape()[1] + j]);
            }
        }

        auto gradients = descriptor.gradients();
        CHECK(gradients.shape() == std::array<size_t, 2>{18, 1});
        expected_data = std::vector<double>{
            1, /**/ 1, /**/ 1, /**/ 1, /**/ 1, /**/ 1,
            1, /**/ 1, /**/ 1, /**/ 1, /**/ 1, /**/ 1,
            1, /**/ 1, /**/ 1, /**/ 1, /**/ 1, /**/ 1,
        };
        for (size_t i=0; i<gradients.shape()[0]; i++) {
            for (size_t j=0; j<gradients.shape()[1]; j++) {
                CHECK(gradients(i, j) == expected_data[i * gradients.shape()[1] + j]);
            }
        }
    }

    SECTION("Partial compute -- errors") {
        auto options = rascaline::CalculationOptions();
        options.selected_samples = rascaline::SelectedIndexes({"structure", "center", "species"});
        options.selected_samples.add({0, 1, 3});

        CHECK_THROWS_WITH(
            calculator.compute(systems, std::move(options)),
            "invalid parameter: 'species' in requested samples is not part of the samples of this calculator"
        );

        options = rascaline::CalculationOptions();
        options.selected_features = rascaline::SelectedIndexes({"index_delta", "x_y_z", "foo"});
        options.selected_features.add({0, 1, 3});

        CHECK_THROWS_WITH(
            calculator.compute(systems, std::move(options)),
            "invalid parameter: 'foo' in requested features is not part of the features of this calculator"
        );
    }
}

static void check_indexes(
    const rascaline::Indexes& indexes,
    std::vector<std::string> names,
    std::array<size_t, 2> shape,
    std::vector<int32_t> values
) {
    CHECK(indexes.names() == names);
    CHECK(indexes.shape() == shape);

    for (size_t i=0; i<indexes.shape()[0]; i++) {
        for (size_t j=0; j<indexes.shape()[1]; j++) {
            CHECK(indexes(i, j) == values[i * indexes.shape()[1] + j]);
        }
    }
}
