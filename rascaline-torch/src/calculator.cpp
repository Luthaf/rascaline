#include "rascaline/torch/calculator.hpp"
#include "rascaline/torch/autograd.hpp"

using namespace rascaline_torch;

// move a block created by rascaline to torch
static equistore::TensorBlock block_to_torch(
    std::shared_ptr<equistore::TensorMap> tensor,
    equistore::TensorBlock block
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

    auto new_block = equistore::TensorBlock(
        std::unique_ptr<equistore::DataArrayBase>(new equistore_torch::TorchDataArray(std::move(torch_values))),
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


equistore_torch::TorchTensorMap CalculatorHolder::compute_impl(
    std::vector<TorchSystem>& systems,
    rascaline::CalculationOptions options
) {
    // convert the systems
    auto base_systems = std::vector<rascal_system_t>();
    base_systems.reserve(systems.size());
    for (auto& system: systems) {
        base_systems.push_back(system->as_rascal_system_t());
    }

    // run the calculations
    auto descriptor = std::make_shared<equistore::TensorMap>(calculator_.compute(base_systems, options));

    // move all data to torch
    auto blocks = std::vector<equistore::TensorBlock>();
    for (size_t block_i=0; block_i<descriptor->keys().count(); block_i++) {
        blocks.push_back(block_to_torch(descriptor, descriptor->block_by_id(block_i)));
    }

    return torch::make_intrusive<equistore_torch::TensorMapHolder>(
        equistore::TensorMap(descriptor->keys(), std::move(blocks))
    );
}


equistore_torch::TorchTensorMap CalculatorHolder::compute(
    std::vector<TorchSystem> systems,
    std::vector<std::string> gradients
) {
    auto all_positions_vec = std::vector<torch::Tensor>();
    all_positions_vec.reserve(systems.size());

    auto all_cells_vec = std::vector<torch::Tensor>();
    all_cells_vec.reserve(systems.size());

    for (const auto& system: systems) {
        all_positions_vec.push_back(system->get_positions());
        all_cells_vec.push_back(system->get_cell());
    }

    auto all_positions = torch::vstack(all_positions_vec);
    auto all_cells = torch::vstack(all_cells_vec);

    auto descriptor = equistore_torch::TorchTensorMap();

    // see `RascalineAutograd::forward` for an explanation of what's happening
    auto _ = RascalineAutograd::apply(
        all_positions,
        all_cells,
        *this,
        std::move(systems),
        &descriptor,
        std::move(gradients)
    );

    return descriptor;
}
