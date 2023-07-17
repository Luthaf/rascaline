
#include <torch/torch.h>

#include <rascaline.hpp>
#include <rascaline/torch.hpp>

#include <catch.hpp>

using namespace rascaline_torch;

static TorchSystem test_system(bool positions_grad, bool cell_grad);


TEST_CASE("Calculator") {
    auto HYPERS_JSON = R"({
        "cutoff": 3.0,
        "delta": 4,
        "name": "bar"
    })";
    auto calculator = CalculatorHolder("dummy_calculator", HYPERS_JSON);

    SECTION("Parameters") {
        CHECK(calculator.name() == "dummy test calculator with cutoff: 3 - delta: 4 - name: bar");

        CHECK(calculator.parameters() == HYPERS_JSON);
    }

    SECTION("Compute -- no gradients") {
        auto system = test_system(false, false);
        auto descriptor = calculator.compute({system});

        CHECK(*descriptor->keys() == equistore::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor->block_by_id(0);
        CHECK(*block->samples() == equistore::Labels(
            {"structure", "center"},
            {{0, 1}, {0, 2}, {0, 3}}
        ));
        CHECK(*block->properties() == equistore::Labels(
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
        block = descriptor->block_by_id(1);
        CHECK(*block->samples() == equistore::Labels(
            {"structure", "center"},
            {{0, 0}}
        ));
        CHECK(*block->properties() == equistore::Labels(
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

    SECTION("Compute -- all gradients") {
        auto system = test_system(true, false);
        auto descriptor = calculator.compute({system}, /* forward_gradients */ {"positions"});

        CHECK(*descriptor->keys() == equistore::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor->block_by_id(0);

        auto values = block->values();
        CHECK(values.requires_grad() == true);

        auto grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK(grad_fn->name() == "torch::autograd::CppNode<rascaline_torch::RascalineAutograd>");

        // forward gradients
        auto gradient = block->gradient("positions");
        CHECK(*gradient->samples() == equistore::Labels(
            {"sample", "structure", "atom"},
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
        block = descriptor->block_by_id(1);

        values = block->values();
        CHECK(values.requires_grad() == true);

        grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK(grad_fn->name() == "torch::autograd::CppNode<rascaline_torch::RascalineAutograd>");

        // forward gradients
        gradient = block->gradient("positions");
        CHECK(*gradient->samples() == equistore::Labels(
            {"sample", "structure", "atom"},
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

        CHECK(*descriptor->keys() == equistore::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor->block_by_id(0);

        auto values = block->values();
        CHECK(values.requires_grad() == true);

        auto grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK(grad_fn->name() == "torch::autograd::CppNode<rascaline_torch::RascalineAutograd>");

        // no forward gradients
        CHECK(block->gradients_list().empty());

        // C block
        block = descriptor->block_by_id(1);

        values = block->values();
        CHECK(values.requires_grad() == true);

        grad_fn = values.grad_fn();
        REQUIRE(grad_fn);
        CHECK(grad_fn->name() == "torch::autograd::CppNode<rascaline_torch::RascalineAutograd>");

        // no forward gradients
        CHECK(block->gradients_list().empty());
    }

    SECTION("Compute -- no backward gradients") {
        auto system = test_system(false, false);
        auto descriptor = calculator.compute({system}, /* forward_gradients */ {"positions"});

        CHECK(*descriptor->keys() == equistore::Labels(
            {"species_center"},
            {{1}, {6}}
        ));

        // H block
        auto block = descriptor->block_by_id(0);

        auto values = block->values();
        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);

        // forward gradients
        auto gradient = block->gradient("positions");
        CHECK(gradient->samples()->count() == 8);

        // C block
        block = descriptor->block_by_id(1);

        values = block->values();
        CHECK(values.requires_grad() == false);
        CHECK(values.grad_fn() == nullptr);

        // forward gradients
        gradient = block->gradient("positions");
        CHECK(gradient->samples()->count() == 2);
    }
}

TorchSystem test_system(bool positions_grad, bool cell_grad) {
    auto species = torch::tensor({6, 1, 1, 1});
    auto positions = torch::tensor({
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
    }).reshape({4, 3});
    positions.requires_grad_(positions_grad);

    auto cell = 10 * torch::eye(3);
    cell.requires_grad_(cell_grad);

    return torch::make_intrusive<SystemHolder>(species, positions, cell);
}
