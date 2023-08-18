#include <vector>
#include <string>

#include "rascaline.hpp"
#include "catch.hpp"

#include "test_system.hpp"

TEST_CASE("Calculator name") {
    SECTION("dummy_calculator") {
        const char* HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar"
        })";
        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

        CHECK(calculator.name() == "dummy test calculator with cutoff: 3.5 - delta: 25 - name: bar");
    }

    SECTION("long strings") {
        auto name = std::string(2048, 'b');
        auto HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": ")" + name + "\"}";

        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

        std::string expected = "dummy test calculator with cutoff: 3.5 - delta: 25 - ";
        expected += "name: " + name;
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

    SECTION("cutoffs") {
        std::string HYPERS_JSON = R"({
            "cutoff": 3.5,
            "delta": 25,
            "name": "bar",
            "gradients": false
        })";
        auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);
        CHECK(calculator.cutoffs() == std::vector<double>{3.5});
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
        "cutoff": 3.0, "delta": 4, "name": ""
    })";

    auto systems = std::vector<TestSystem>{TestSystem()};
    auto calculator = rascaline::Calculator("dummy_calculator", HYPERS_JSON);

    SECTION("Full compute") {
        auto options = rascaline::CalculationOptions();
        options.gradients.push_back("positions");
        auto descriptor = calculator.compute(systems, options);

        CHECK(descriptor.keys() == metatensor::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor.block_by_id(0);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 1}, {0, 2}, {0, 3}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {5.0, 39.0, 6.0, 18.0, 7.0, 15.0},
            {3, 2}
        ));

        auto gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {
                {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
                {1, 0, 1}, {1, 0, 2}, {1, 0, 3},
                {2, 0, 2}, {2, 0, 3},
            }
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            },
            {8, 3, 2}
        ));

        // C block
        block = descriptor.block_by_id(1);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 0}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {4.0, 33.0},
            {1, 2}
        ));

        gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {{0, 0, 0}, {0, 0, 1}}
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            },
            {2, 3, 2}
        ));
    }

    SECTION("Partial compute -- samples") {
        auto options = rascaline::CalculationOptions();
        options.gradients.push_back("positions");
        options.selected_samples = rascaline::LabelsSelection::subset(
            metatensor::Labels({"structure", "center"}, {{0, 1}, {0, 3}})
        );
        auto descriptor = calculator.compute(systems, options);

        CHECK(descriptor.keys() == metatensor::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor.block_by_id(0);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 1}, {0, 3}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {5.0, 39.0, 7.0, 15.0},
            {2, 2}
        ));

        auto gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {
                {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
                {1, 0, 2}, {1, 0, 3},
            }
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
                0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            },
            {5, 3, 2}
        ));

        // C block
        block = descriptor.block_by_id(1);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            std::vector<double>{},
            {0, 2}
        ));

        gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {}
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            std::vector<double>{},
            {0, 3, 2}
        ));
    }

    SECTION("Partial compute -- features") {
        auto options = rascaline::CalculationOptions();
        options.gradients.push_back("positions");
        options.selected_properties = rascaline::LabelsSelection::subset(
            metatensor::Labels({"index_delta", "x_y_z"}, {{0, 1}})
        );
        auto descriptor = calculator.compute(systems, options);

        CHECK(descriptor.keys() == metatensor::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor.block_by_id(0);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 1}, {0, 2}, {0, 3}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {39.0, 18.0, 15.0},
            {3, 1}
        ));

        auto gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {
                {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
                {1, 0, 1}, {1, 0, 2}, {1, 0, 3},
                {2, 0, 2}, {2, 0, 3},
            }
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
            },
            {8, 3, 1}
        ));

        // C block
        block = descriptor.block_by_id(1);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 0}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {33.0},
            {1, 1}
        ));

        gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {{0, 0, 0}, {0, 0, 1}}
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
            },
            {2, 3, 1}
        ));
    }

    SECTION("Partial compute -- preselected") {
        auto options = rascaline::CalculationOptions();
        options.gradients.push_back("positions");
        auto blocks = std::vector<metatensor::TensorBlock>();

        blocks.emplace_back(
            metatensor::TensorBlock(
                std::unique_ptr<metatensor::SimpleDataArray>(new metatensor::SimpleDataArray({1, 1})),
                metatensor::Labels({"structure", "center"}, {{0, 3}}),
                {},
                metatensor::Labels({"index_delta", "x_y_z"}, {{0, 1}})
            )
        );

        blocks.emplace_back(
            metatensor::TensorBlock(
                std::unique_ptr<metatensor::SimpleDataArray>(new metatensor::SimpleDataArray({1, 1})),
                metatensor::Labels({"structure", "center"}, {{0, 0}}),
                {},
                metatensor::Labels({"index_delta", "x_y_z"}, {{1, 0}})
            )
        );

        auto predefined = std::make_shared<metatensor::TensorMap>(
            metatensor::Labels({"species_center"}, {{1}, {6}}),
            std::move(blocks)
        );
        options.selected_samples = rascaline::LabelsSelection::predefined(predefined);
        options.selected_properties = rascaline::LabelsSelection::predefined(predefined);

        auto descriptor = calculator.compute(systems, options);

        CHECK(descriptor.keys() == metatensor::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor.block_by_id(0);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 3}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {15.0},
            {1, 1}
        ));

        auto gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {
                {0, 0, 2}, {0, 0, 3},
            }
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                1.0, /**/ 1.0, /**/ 1.0,
                1.0, /**/ 1.0, /**/ 1.0,
            },
            {2, 3, 1}
        ));

        // C block
        block = descriptor.block_by_id(1);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 0}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {4.0},
            {1, 1}
        ));

        gradient = block.gradient("positions");
        CHECK(gradient.samples() == metatensor::Labels(
            {"sample", "structure", "atom"},
            {{0, 0, 0}, {0, 0, 1}}
        ));
        CHECK(gradient.values() == metatensor::NDArray<double>(
            {
                0.0, /**/ 0.0, /**/ 0.0,
                0.0, /**/ 0.0, /**/ 0.0,
            },
            {2, 3, 1}
        ));
    }

    SECTION("Partial compute -- key selection") {
        // check key selection: we add a non-existing key (12) and remove an
        // existing one (1) from the default set of keys. We also put the keys
        // in a different order than what would be the default (6, 12).

        auto options = rascaline::CalculationOptions();
        options.selected_keys = metatensor::Labels(
            {"species_center"},
            {{12}, {6}}
        );
        auto descriptor = calculator.compute(systems, options);

        CHECK(descriptor.keys() == metatensor::Labels(
            {"species_center"},
            {{12}, {6}}
        ));

        // empty block
        auto block = descriptor.block_by_id(0);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            std::vector<double>{},
            {0, 2}
        ));

        // C block
        block = descriptor.block_by_id(1);
        CHECK(block.samples() == metatensor::Labels(
            {"structure", "center"},
            {{0, 0}}
        ));
        CHECK(block.properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));
        CHECK(block.values() == metatensor::NDArray<double>(
            {4.0, 33.0},
            {1, 2}
        ));
    }
}
