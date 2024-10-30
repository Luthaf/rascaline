#include <torch/torch.h>

#include <rascaline.hpp>
#include <metatensor/torch.hpp>

#include "metatensor/torch/labels.hpp"
#include "rascaline/torch.hpp"

#include <catch.hpp>

using namespace rascaline_torch;
using namespace metatensor_torch;

static metatensor_torch::System test_system(bool positions_grad, bool cell_grad);


TEST_CASE("Calculator") {
    const auto* HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "bar"
    })";
    auto calculator = CalculatorHolder("dummy_calculator", HYPERS_JSON);

    SECTION("Parameters") {
        CHECK(calculator.name() == "dummy test calculator with cutoff: 3 - delta: 4 - name: bar");
        CHECK(calculator.parameters() == HYPERS_JSON);
        CHECK(calculator.cutoffs() == std::vector<double>{3.0});
    }

    SECTION("Compute -- no gradients") {
        auto system = test_system(false, false);
        auto descriptor = calculator.compute({system});

        CHECK(*descriptor->keys() == metatensor::Labels(
            {"center_type"},
            {{1}, {6}}
        ));

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);
        CHECK(*block->samples() == metatensor::Labels(
            {"system", "atom"},
            {{0, 1}, {0, 2}, {0, 3}}
        ));
        CHECK(*block->properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));

        auto expected = torch::tensor({5.0, 9.0, 6.0, 18.0, 7.0, 15.0}).reshape({3, 2});
        auto values = block->values();
        CHECK(torch::all(values == expected).item<bool>());

        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);

        // no gradient requested
        CHECK(block->gradients_list().empty());

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);
        CHECK(*block->samples() == metatensor::Labels(
            {"system", "atom"},
            {{0, 0}}
        ));
        CHECK(*block->properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}, {0, 1}}
        ));

        expected = torch::tensor({4.0, 3.0}).reshape({1, 2});
        values = block->values();
        CHECK(torch::all(values == expected).item<bool>());

        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);
        CHECK(block->gradients_list().empty());
    }

    SECTION("keys selection") {
        auto system = test_system(false, false);

        auto options = torch::make_intrusive<CalculatorOptionsHolder>();
        options->set_selected_keys(LabelsHolder::create({"center_type"}, {{12}, {1}}));
        auto descriptor = calculator.compute({system}, options);

        CHECK(*descriptor->keys() == metatensor::Labels(
            {"center_type"},
            {{12}, {1}}
        ));

        // empty block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);
        auto values = block->values();
        CHECK(values.sizes() == std::vector<int64_t>{0, 2});

        // H block
        block = TensorMapHolder::block_by_id(descriptor, 1);
        auto expected = torch::tensor({5.0, 9.0, 6.0, 18.0, 7.0, 15.0}).reshape({3, 2});
        values = block->values();
        CHECK(torch::all(values == expected).item<bool>());
    }

    SECTION("sample selection") {
        auto system = test_system(false, false);

        auto options = torch::make_intrusive<CalculatorOptionsHolder>();
        options->set_selected_samples(LabelsHolder::create({"atom"}, {{0}, {2}}));
        auto descriptor = calculator.compute({system}, options);

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);
        CHECK(*block->samples() == metatensor::Labels(
            {"system", "atom"},
            {{0, 2}}
        ));

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);
        CHECK(*block->samples() == metatensor::Labels(
            {"system", "atom"},
            {{0, 0}}
        ));
    }

    SECTION("properties selection") {
        auto system = test_system(false, false);

        auto options = torch::make_intrusive<CalculatorOptionsHolder>();
        options->set_selected_properties(LabelsHolder::create({"index_delta"}, {{1}, {12}}));
        auto descriptor = calculator.compute({system}, options);

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);
        CHECK(*block->properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}}
        ));

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);
        CHECK(*block->properties() == metatensor::Labels(
            {"index_delta", "x_y_z"},
            {{1, 0}}
        ));
    }


    SECTION("Compute -- all gradients") {
        auto system = test_system(true, false);

        auto options = torch::make_intrusive<CalculatorOptionsHolder>();
        options->gradients.emplace_back("positions");
        auto descriptor = calculator.compute({system}, options);

        CHECK(*descriptor->keys() == metatensor::Labels(
            {"center_type"},
            {{1}, {6}}
        ));

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);

        auto values = block->values();
        CHECK(values.requires_grad() == true);

        auto grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK_THAT(grad_fn->name(), Catch::Matchers::Contains("rascaline_torch::RascalineAutograd"));

        // forward gradients
        auto gradient = TensorBlockHolder::gradient(block, "positions");
        CHECK(*gradient->samples() == metatensor::Labels(
            {"sample", "system", "atom"},
            {
                {0, 0, 0}, {0, 0, 1}, {0, 0, 2},
                {1, 0, 1}, {1, 0, 2}, {1, 0, 3},
                {2, 0, 2}, {2, 0, 3},
            }
        ));
        auto expected = torch::tensor({
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
        }).reshape({8, 3, 2});
        CHECK(torch::all(gradient->values() == expected).item<bool>());

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);

        values = block->values();
        CHECK(values.requires_grad() == true);

        grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK_THAT(grad_fn->name(), Catch::Matchers::Contains("rascaline_torch::RascalineAutograd"));

        // forward gradients
        gradient = TensorBlockHolder::gradient(block, "positions");
        CHECK(*gradient->samples() == metatensor::Labels(
            {"sample", "system", "atom"},
            {{0, 0, 0}, {0, 0, 1}}
        ));
        expected = torch::tensor({
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
            0.0, 1.0, /**/ 0.0, 1.0, /**/ 0.0, 1.0,
        }).reshape({2, 3, 2});
        CHECK(torch::all(gradient->values() == expected).item<bool>());
    }

    SECTION("Compute -- no forward gradients") {
        auto system = test_system(true, false);
        auto descriptor = calculator.compute({system}, /* forward_gradients */ {});

        CHECK(*descriptor->keys() == metatensor::Labels(
            {"center_type"},
            {{1}, {6}}
        ));

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);

        auto values = block->values();
        CHECK(values.requires_grad() == true);

        auto grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK_THAT(grad_fn->name(), Catch::Matchers::Contains("rascaline_torch::RascalineAutograd"));

        // no forward gradients
        CHECK(block->gradients_list().empty());

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);

        values = block->values();
        CHECK(values.requires_grad() == true);

        grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK_THAT(grad_fn->name(), Catch::Matchers::Contains("rascaline_torch::RascalineAutograd"));

        // no forward gradients
        CHECK(block->gradients_list().empty());
    }

    SECTION("Compute -- no backward gradients") {
        auto system = test_system(false, false);

        auto options = torch::make_intrusive<CalculatorOptionsHolder>();
        options->gradients.emplace_back("positions");
        auto descriptor = calculator.compute({system}, options);

        CHECK(*descriptor->keys() == metatensor::Labels(
            {"center_type"},
            {{1}, {6}}
        ));

        // H block
        auto block = TensorMapHolder::block_by_id(descriptor, 0);

        auto values = block->values();
        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);

        // forward gradients
        auto gradient = TensorBlockHolder::gradient(block, "positions");
        CHECK(gradient->samples()->count() == 8);

        // C block
        block = TensorMapHolder::block_by_id(descriptor, 1);

        values = block->values();
        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);

        // forward gradients
        gradient = TensorBlockHolder::gradient(block, "positions");
        CHECK(gradient->samples()->count() == 2);
    }
}

metatensor_torch::System test_system(bool positions_grad, bool cell_grad) {
    auto types = torch::tensor({6, 1, 1, 1});
    auto positions = torch::tensor({
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
    }).reshape({4, 3});
    positions.requires_grad_(positions_grad);

    auto cell = 10 * torch::eye(3);
    cell.requires_grad_(cell_grad);

    auto pbc = torch::ones(3, torch::TensorOptions().dtype(torch::kBool));

    return torch::make_intrusive<metatensor_torch::SystemHolder>(types, positions, cell, pbc);
}
