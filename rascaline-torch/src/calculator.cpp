#include "rascaline/torch/calculator.hpp"
#include "metatensor/torch/tensor.hpp"
#include "rascaline/torch/autograd.hpp"
#include <c10/util/Exception.h>

using namespace metatensor_torch;
using namespace rascaline_torch;

// move a block created by rascaline to torch
static metatensor::TensorBlock block_to_torch(
    std::shared_ptr<metatensor::TensorMap> tensor,
    metatensor::TensorBlock block
) {
    auto values = block.values();
    auto sizes = std::vector<int64_t>();
    for (auto size: values.shape()) {
        sizes.push_back(static_cast<int64_t>(size));
    }

    auto torch_values = torch::from_blob(
        values.data(),
        std::move(sizes),
        [tensor](void*){
            // this function holds a copy of `tensor`, which will make sure that
            // (a) the TensorMap is kept alive for as long as the values
            // returned by `torch::from_blob` is; and (b) the TensorMap will be
            // freed once all the torch::Tensor created with `torch::from_blob`
            // are freed as well
            auto _ = std::move(tensor);
        },
        torch::TensorOptions().dtype(torch::kF64).device(torch::kCPU)
    );

    auto new_block = metatensor::TensorBlock(
        std::unique_ptr<metatensor::DataArrayBase>(new metatensor_torch::TorchDataArray(std::move(torch_values))),
        block.samples(),
        block.components(),
        block.properties()
    );

    for (auto parameter: block.gradients_list()) {
        auto gradient = block_to_torch(tensor, block.gradient(parameter));
        new_block.add_gradient(std::move(parameter), std::move(gradient));
    }

    return new_block;
}

static torch::Tensor stack_all_positions(const std::vector<TorchSystem>& systems) {
    auto all_positions = std::vector<torch::Tensor>();
    all_positions.reserve(systems.size());

    for (const auto& system: systems) {
        all_positions.push_back(system->get_positions());
    }

    return torch::vstack(all_positions);
}

static torch::Tensor stack_all_cells(const std::vector<TorchSystem>& systems) {
    auto all_cells = std::vector<torch::Tensor>();
    all_cells.reserve(systems.size());

    for (const auto& system: systems) {
        all_cells.push_back(system->get_cell());
    }

    return torch::vstack(all_cells);
}

static bool all_systems_use_native(const std::vector<TorchSystem>& systems) {
    auto result = systems[0]->use_native_system();
    for (const auto& system: systems) {
        if (system->use_native_system() != result) {
            C10_THROW_ERROR(ValueError,
                "either all or none of the systems should have pre-defined neighbor lists"
            );
        }
    }
    return result;
}

static TorchTensorMap remove_other_gradients(
    TorchTensorMap tensor,
    const std::vector<std::string>& gradients_to_keep
) {
    auto new_blocks = std::vector<TorchTensorBlock>();
    for (int64_t i=0; i<tensor->keys()->count(); i++) {
        auto block = tensor->block_by_id(i);
        auto new_block = torch::make_intrusive<TensorBlockHolder>(
            block->values(),
            block->samples(),
            block->components(),
            block->properties()
        );

        for (const auto& parameter: gradients_to_keep) {
            auto gradient = block->gradient(parameter);
            new_block->add_gradient(parameter, gradient);
        }

        new_blocks.push_back(std::move(new_block));
    }

    return torch::make_intrusive<TensorMapHolder>(
        tensor->keys(),
        std::move(new_blocks)
    );
}

static bool contains(const std::vector<std::string>& haystack, const std::string& needle) {
    return std::find(std::begin(haystack), std::end(haystack), needle) != std::end(haystack);
}


metatensor_torch::TorchTensorMap CalculatorHolder::compute(
    std::vector<TorchSystem> systems,
    std::vector<std::string> gradients
) {
    auto all_positions = stack_all_positions(systems);
    auto all_cells = stack_all_cells(systems);
    auto structures_start_ivalue = torch::IValue();

    // =============== Handle all options for the calculation =============== //
    auto options = rascaline::CalculationOptions();

    // which gradients should we compute? We have to compute some gradient
    // either if positions/cell has `requires_grad` set to `true`, or if the
    // user requested specific gradients in `forward_gradients`
    for (const auto& parameter: gradients) {
        if (parameter != "positions" && parameter != "cell") {
            C10_THROW_ERROR(ValueError, "invalid gradients requested: " + parameter);
        }
    }

    if (contains(gradients, "positions") || all_positions.requires_grad()) {
        options.gradients.push_back("positions");

        auto structures_start = c10::List<int64_t>();
        int64_t current_start = 0;
        for (auto& system: systems) {
            structures_start.push_back(current_start);
            current_start += system->size();
        }
        structures_start_ivalue = torch::IValue(std::move(structures_start));
    }

    if (contains(gradients, "cell") || all_cells.requires_grad()) {
        options.gradients.push_back("cell");
    }

    // where all computed gradients explicitly requested in forward_gradients?
    bool all_forward_gradients = true;
    for (const auto& parameter: options.gradients) {
        if (!contains(gradients, parameter)) {
            all_forward_gradients = false;
        }
    }

    options.use_native_system = all_systems_use_native(systems);
    // TODO: selected_properties
    // TODO: selected_samples

    // convert the systems
    auto base_systems = std::vector<rascal_system_t>();
    base_systems.reserve(systems.size());
    for (auto& system: systems) {
        base_systems.push_back(system->as_rascal_system_t());
    }

    // ============ run the calculation and move data to torch ============== //
    auto raw_descriptor = std::make_shared<metatensor::TensorMap>(
        calculator_.compute(base_systems, options)
    );

    // move all data to torch
    auto blocks = std::vector<metatensor::TensorBlock>();
    for (size_t block_i=0; block_i<raw_descriptor->keys().count(); block_i++) {
        blocks.push_back(block_to_torch(raw_descriptor, raw_descriptor->block_by_id(block_i)));
    }

    auto torch_descriptor = torch::make_intrusive<metatensor_torch::TensorMapHolder>(
        metatensor::TensorMap(raw_descriptor->keys(), std::move(blocks))
    );

    // ============ register the autograd nodes for each block ============== //
    auto all_positions_vec = std::vector<torch::Tensor>();
    all_positions_vec.reserve(systems.size());

    auto all_cells_vec = std::vector<torch::Tensor>();
    all_cells_vec.reserve(systems.size());

    for (const auto& system: systems) {
        all_positions_vec.push_back(system->get_positions());
        all_cells_vec.push_back(system->get_cell());
    }

    for (int64_t block_i=0; block_i<torch_descriptor->keys()->count(); block_i++) {
        auto block = torch_descriptor->block_by_id(block_i);
        // see `RascalineAutograd::forward` for an explanation of what's happening
        auto _ = RascalineAutograd::apply(
            all_positions,
            all_cells,
            structures_start_ivalue,
            block
        );
    }

    // ====================== handle forward gradients ====================== //
    if (all_forward_gradients) {
        return torch_descriptor;
    } else {
        return remove_other_gradients(torch_descriptor, gradients);
    }
}


metatensor_torch::TorchTensorMap rascaline_torch::register_autograd(
    std::vector<TorchSystem> systems,
    metatensor_torch::TorchTensorMap precomputed,
    std::vector<std::string> forward_gradients
) {
    if (precomputed->keys()->count() == 0) {
        return precomputed;
    }

    auto all_positions = stack_all_positions(systems);
    auto all_cells = stack_all_cells(systems);
    auto structures_start_ivalue = torch::IValue();

    auto precomputed_gradients = precomputed->block_by_id(0)->gradients_list();

    if (all_positions.requires_grad()) {
        if (!contains(precomputed_gradients, "positions")) {
            C10_THROW_ERROR(ValueError,
                "expected the precomputed TensorMap to contain gradients with "
                "respect to 'positions' since one of the system `requires_grad` "
                "for its positions"
            );
        }

        auto structures_start = c10::List<int64_t>();
        int64_t current_start = 0;
        for (auto& system: systems) {
            structures_start.push_back(current_start);
            current_start += system->size();
        }
        structures_start_ivalue = torch::IValue(std::move(structures_start));
    }

    if (all_cells.requires_grad()) {
        if (!contains(precomputed_gradients, "cell")) {
            C10_THROW_ERROR(ValueError,
                "expected the precomputed TensorMap to contain gradients with "
                "respect to 'cell' since one of the system `requires_grad` "
                "for its cell"
            );
        }
    }

    // Does `forward_gradients` contains the same gradients as `precomputed_gradients`?
    bool all_forward_gradients = true;
    if (forward_gradients.size() != precomputed_gradients.size()) {
        all_forward_gradients = false;
    } else {
        for (const auto& parameter: forward_gradients) {
            if (!contains(precomputed_gradients, parameter)) {
                all_forward_gradients = false;
            }
        }
    }

    for (int64_t block_i=0; block_i<precomputed->keys()->count(); block_i++) {
        auto block = precomputed->block_by_id(block_i);
        auto _ = RascalineAutograd::apply(
            all_positions,
            all_cells,
            structures_start_ivalue,
            block
        );
    }

    // ====================== handle forward gradients ====================== //
    if (all_forward_gradients) {
        return precomputed;
    } else {
        return remove_other_gradients(precomputed, forward_gradients);
    }
}
