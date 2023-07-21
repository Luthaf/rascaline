#include <algorithm>

#include "rascaline/torch/autograd.hpp"

using namespace rascaline_torch;

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

static bool contains(const std::vector<std::string>& haystack, const std::string& needle) {
    return std::find(std::begin(haystack), std::end(haystack), needle) != std::end(haystack);
}

std::vector<torch::Tensor> EquistoreAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor all_positions,
    torch::Tensor all_cells,
    equistore_torch::TorchTensorMap descriptor,
    std::vector<TorchSystem> systems
) {
    // =============== Handle all options for the calculation =============== //
    auto calculation_options = rascaline::CalculationOptions();

    // which gradients should we compute? We have to compute some gradient
    // either if positions/cell has `requires_grad` set to `true`, or if the
    // user requested specific gradients in `forward_gradients`
    
    
    // PR COMMENT change to use gradients in descriptor?
    //for (const auto& parameter: forward_gradients) {
    //    if (parameter != "positions" && parameter != "cell") {
    //        C10_THROW_ERROR(ValueError, "invalid parameter in forward gradients: " + parameter);
    //    }
    //}
    //if (contains(forward_gradients, "positions") || all_positions.requires_grad()) {
    //    calculation_options.gradients.push_back("positions");
    //}

    //if (contains(forward_gradients, "cell") || all_cells.requires_grad()) {
    //    calculation_options.gradients.push_back("cell");
    //}

    // PR COMMENT This option can be probably removed
    // where all computed gradients explicitly requested in forward_gradients?
    //bool all_forward_gradients = true;
    //for (const auto& parameter: calculation_options.gradients) {
    //    if (!contains(forward_gradients, parameter)) {
    //        all_forward_gradients = false;
    //    }
    //}

    calculation_options.use_native_system = all_systems_use_native(systems);
    // TODO: selected_properties
    // TODO: selected_samples

    // =================== run the actual calculation ======================= //
    auto structures_start = std::vector<int64_t>();
    int64_t current_start = 0;
    for (auto& system: systems) {
        structures_start.push_back(current_start);
        current_start += system->size();
    }

    // ================== extract the data for autograd ===================== //
    auto values_by_block = std::vector<torch::Tensor>();
    for (int64_t i=0; i< descriptor->keys()->count(); i++) {
        // this add a reference to the torch::Tensor already in `descriptor`
        // inside `values_by_block`
        values_by_block.push_back(descriptor->block_by_id(i)->values());
    }

    // ============== save the required data for backward pass ============== //
    ctx->save_for_backward({all_positions, all_cells});
    ctx->saved_data["descriptor"] = descriptor;
    ctx->saved_data["structures_start"] = structures_start;


    // ==================== "return" the right TensorMap ==================== //
    // PR COMMENT not needed I think
    //if (all_forward_gradients) {
    //    *tensor_map = std::move(descriptor);
    //} else {
    //    // create a new TensorMap with only the requested forward gradients
    //    auto new_blocks = std::vector<equistore_torch::TorchTensorBlock>();
    //    for (int64_t i=0; i<descriptor->keys()->count(); i++) {
    //        auto block = descriptor->block_by_id(i);

    //        auto new_block = torch::make_intrusive<equistore_torch::TensorBlockHolder>(
    //            block->values(),
    //            block->samples(),
    //            block->components(),
    //            block->properties()
    //        );

    //        for (const auto& parameter: forward_gradients) {
    //            auto gradient = block->gradient(parameter);
    //            new_block->add_gradient(parameter, gradient);
    //        }

    //        new_blocks.push_back(std::move(new_block));
    //    }
    //    *tensor_map = torch::make_intrusive<equistore_torch::TensorMapHolder>(
    //        descriptor->keys(),
    //        std::move(new_blocks)
    //    );
    //}

    return values_by_block;
}

torch::autograd::variable_list EquistoreAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    // ============== get the saved data from the forward pass ============== //
    auto saved_variables = ctx->get_saved_variables();
    auto all_positions = saved_variables[0];
    auto all_cells = saved_variables[1];

    auto descriptor = ctx->saved_data["descriptor"].toCustomClass<equistore_torch::TensorMapHolder>();
    auto structures_start = ctx->saved_data["structures_start"].toIntVector();

    auto n_blocks = grad_outputs.size();
    assert(descriptor->keys()->count() == n_blocks);

    // ===================== gradient w.r.t. positions ====================== //
    auto positions_grad = torch::Tensor();
    if (all_positions.requires_grad()) {
        positions_grad = torch::zeros_like(all_positions);
        assert(positions_grad.is_contiguous() && positions_grad.is_cpu());
        auto positions_grad_ptr = positions_grad.data_ptr<double>();

        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            auto gradient = descriptor->block_by_id(block_i)->gradient("positions");

            auto samples = gradient->samples();
            auto samples_values = samples->values();
            auto samples_values_ptr = samples_values.data_ptr<int32_t>();
            assert(samples->names().size() == 3);
            assert(samples->names()[0] == "sample");
            assert(samples->names()[0] == "structure");
            assert(samples->names()[0] == "atom");

            // This is dX/ dr_i, which we computed in the forward pass
            auto forward_values = gradient->values();
            assert(forward_values.is_contiguous() && forward_values.is_cpu());
            auto forward_grad_ptr = forward_values.data_ptr<double>();

            // This is dA/ dX, which torch computed in the beginning of the
            // backward pass
            auto grad_output = grad_outputs[block_i];
            const auto& grad_output_sizes = grad_output.sizes();
            assert(grad_output.is_contiguous() && grad_output.is_cpu());
            auto grad_values_ptr = grad_output.data_ptr<double>();

            // total size of component + property dimension
            int64_t dot_dimensions = 1;
            for (int i=1; i<grad_output_sizes.size(); i++) {
                dot_dimensions *= grad_output_sizes[i];
            }

            // We want to compute dA/dr_i using the data above to finish the
            // backward pass
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = samples_values_ptr[grad_sample_i * 3 + 0];
                auto structure_i = samples_values_ptr[grad_sample_i * 3 + 1];
                auto atom_i = samples_values_ptr[grad_sample_i* 3 + 2];

                auto global_atom_i = structures_start[structure_i] + atom_i;

                for (int64_t direction=0; direction<3; direction++) {
                    auto dA_dr = 0.0;
                    for (int64_t i=0; i<dot_dimensions; i++) {
                        auto dX_dr = forward_grad_ptr[(grad_sample_i * 3 + direction) * dot_dimensions + i];
                        auto dA_dX = grad_values_ptr[sample_i * dot_dimensions + i];
                        dA_dr += dX_dr * dA_dX;
                    }
                    positions_grad_ptr[global_atom_i * 3 + direction] += dA_dr;
                }
            }
        }
    }

    // ======================= gradient w.r.t. cell ========================= //
    auto cell_grad = torch::Tensor();
    if (all_cells.requires_grad()) {
        cell_grad = torch::zeros_like(all_cells);
        assert(cell_grad.is_contiguous() && cell_grad.is_cpu());
        auto cell_grad_ptr = cell_grad.data_ptr<double>();

        // find the index of the "structure" dimension in the samples
        auto sample_names = descriptor->sample_names();
        auto structure_dimension_it = std::find(
            std::begin(sample_names),
            std::end(sample_names),
            "structure"
        );
        if (structure_dimension_it == std::end(sample_names)) {
            C10_THROW_ERROR(ValueError,
                "could not find 'structure' in the samples, this calculator is missing it"
            );
        }
        int64_t structure_dimension = std::distance(std::begin(sample_names), structure_dimension_it);

        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            auto block = descriptor->block_by_id(block_i);
            auto gradient = block->gradient("cell");

            auto structures = block->samples()->values().index({torch::indexing::Slice(), structure_dimension});

            auto samples = gradient->samples();
            auto samples_values = samples->values();
            auto samples_values_ptr = samples_values.data_ptr<int32_t>();
            assert(samples->names().size() == 1);
            assert(samples->names()[0] == "sample");

            // This is dX/ dH, which we computed in the forward pass
            auto forward_values = gradient->values();
            assert(forward_values.is_contiguous() && forward_values.is_cpu());
            auto forward_grad_ptr = forward_values.data_ptr<double>();

            // This is dA/ dX, which torch computed in the beginning of the
            // backward pass
            auto grad_output = grad_outputs[block_i];
            const auto& grad_output_sizes = grad_output.sizes();
            assert(grad_output.is_contiguous() && grad_output.is_cpu());
            auto grad_values_ptr = grad_output.data_ptr<double>();

            // total size of component + property dimension
            int64_t dot_dimensions = 1;
            for (int i=1; i<grad_output_sizes.size(); i++) {
                dot_dimensions *= grad_output_sizes[i];
            }

            // We want to compute dA/dH using the data above to finish the
            // backward pass
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = samples_values_ptr[grad_sample_i];
                // we get the structure from the samples of the values
                auto structure_i = structures[sample_i].item<int32_t>();

                for (int64_t direction_1=0; direction_1<3; direction_1++) {
                    for (int64_t direction_2=0; direction_2<3; direction_2++) {
                        auto dA_dH = 0.0;
                        for (int64_t i=0; i<dot_dimensions; i++) {
                            auto id = (grad_sample_i * 3 + direction_2) * 3 + direction_1;
                            auto dX_dH = forward_grad_ptr[id * dot_dimensions + i];
                            auto dA_dX = grad_values_ptr[sample_i * dot_dimensions + i];
                            dA_dH += dX_dH * dA_dX;
                        }
                        cell_grad_ptr[(structure_i * 3 + direction_1) * 3 + direction_2] += dA_dH;
                    }
                }
            }
        }
    }

    return {
        positions_grad,
        cell_grad,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
}

std::vector<torch::Tensor> RascalineAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor all_positions,
    torch::Tensor all_cells,
    CalculatorHolder& calculator,
    std::vector<TorchSystem> systems,
    equistore_torch::TorchTensorMap* tensor_map,
    std::vector<std::string> forward_gradients
) {
    // =============== Handle all options for the calculation =============== //
    auto calculation_options = rascaline::CalculationOptions();

    // which gradients should we compute? We have to compute some gradient
    // either if positions/cell has `requires_grad` set to `true`, or if the
    // user requested specific gradients in `forward_gradients`
    for (const auto& parameter: forward_gradients) {
        if (parameter != "positions" && parameter != "cell") {
            C10_THROW_ERROR(ValueError, "invalid parameter in forward gradients: " + parameter);
        }
    }

    if (contains(forward_gradients, "positions") || all_positions.requires_grad()) {
        calculation_options.gradients.push_back("positions");
    }

    if (contains(forward_gradients, "cell") || all_cells.requires_grad()) {
        calculation_options.gradients.push_back("cell");
    }

    // where all computed gradients explicitly requested in forward_gradients?
    bool all_forward_gradients = true;
    for (const auto& parameter: calculation_options.gradients) {
        if (!contains(forward_gradients, parameter)) {
            all_forward_gradients = false;
        }
    }

    calculation_options.use_native_system = all_systems_use_native(systems);
    // TODO: selected_properties
    // TODO: selected_samples

    // =================== run the actual calculation ======================= //
    auto structures_start = std::vector<int64_t>();
    int64_t current_start = 0;
    for (auto& system: systems) {
        structures_start.push_back(current_start);
        current_start += system->size();
    }

    auto descriptor = calculator.compute_impl(systems, calculation_options);

    // ================== extract the data for autograd ===================== //
    auto values_by_block = std::vector<torch::Tensor>();
    for (int64_t i=0; i<descriptor->keys()->count(); i++) {
        // this add a reference to the torch::Tensor already in `descriptor`
        // inside `values_by_block`
        values_by_block.push_back(descriptor->block_by_id(i)->values());
    }

    // ============== save the required data for backward pass ============== //
    ctx->save_for_backward({all_positions, all_cells});
    ctx->saved_data["descriptor"] = descriptor;
    ctx->saved_data["structures_start"] = structures_start;


    // ==================== "return" the right TensorMap ==================== //
    if (all_forward_gradients) {
        *tensor_map = std::move(descriptor);
    } else {
        // create a new TensorMap with only the requested forward gradients
        auto new_blocks = std::vector<equistore_torch::TorchTensorBlock>();
        for (int64_t i=0; i<descriptor->keys()->count(); i++) {
            auto block = descriptor->block_by_id(i);

            auto new_block = torch::make_intrusive<equistore_torch::TensorBlockHolder>(
                block->values(),
                block->samples(),
                block->components(),
                block->properties()
            );

            for (const auto& parameter: forward_gradients) {
                auto gradient = block->gradient(parameter);
                new_block->add_gradient(parameter, gradient);
            }

            new_blocks.push_back(std::move(new_block));
        }
        *tensor_map = torch::make_intrusive<equistore_torch::TensorMapHolder>(
            descriptor->keys(),
            std::move(new_blocks)
        );
    }

    return values_by_block;
}

torch::autograd::variable_list RascalineAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    // ============== get the saved data from the forward pass ============== //
    auto saved_variables = ctx->get_saved_variables();
    auto all_positions = saved_variables[0];
    auto all_cells = saved_variables[1];

    auto descriptor = ctx->saved_data["descriptor"].toCustomClass<equistore_torch::TensorMapHolder>();
    auto structures_start = ctx->saved_data["structures_start"].toIntVector();

    auto n_blocks = grad_outputs.size();
    assert(descriptor->keys()->count() == n_blocks);

    // ===================== gradient w.r.t. positions ====================== //
    auto positions_grad = torch::Tensor();
    if (all_positions.requires_grad()) {
        positions_grad = torch::zeros_like(all_positions);
        assert(positions_grad.is_contiguous() && positions_grad.is_cpu());
        auto positions_grad_ptr = positions_grad.data_ptr<double>();

        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            auto gradient = descriptor->block_by_id(block_i)->gradient("positions");

            auto samples = gradient->samples();
            auto samples_values = samples->values();
            auto samples_values_ptr = samples_values.data_ptr<int32_t>();
            assert(samples->names().size() == 3);
            assert(samples->names()[0] == "sample");
            assert(samples->names()[0] == "structure");
            assert(samples->names()[0] == "atom");

            // This is dX/ dr_i, which we computed in the forward pass
            auto forward_values = gradient->values();
            assert(forward_values.is_contiguous() && forward_values.is_cpu());
            auto forward_grad_ptr = forward_values.data_ptr<double>();

            // This is dA/ dX, which torch computed in the beginning of the
            // backward pass
            auto grad_output = grad_outputs[block_i];
            const auto& grad_output_sizes = grad_output.sizes();
            assert(grad_output.is_contiguous() && grad_output.is_cpu());
            auto grad_values_ptr = grad_output.data_ptr<double>();

            // total size of component + property dimension
            int64_t dot_dimensions = 1;
            for (int i=1; i<grad_output_sizes.size(); i++) {
                dot_dimensions *= grad_output_sizes[i];
            }

            // We want to compute dA/dr_i using the data above to finish the
            // backward pass
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = samples_values_ptr[grad_sample_i * 3 + 0];
                auto structure_i = samples_values_ptr[grad_sample_i * 3 + 1];
                auto atom_i = samples_values_ptr[grad_sample_i* 3 + 2];

                auto global_atom_i = structures_start[structure_i] + atom_i;

                for (int64_t direction=0; direction<3; direction++) {
                    auto dA_dr = 0.0;
                    for (int64_t i=0; i<dot_dimensions; i++) {
                        auto dX_dr = forward_grad_ptr[(grad_sample_i * 3 + direction) * dot_dimensions + i];
                        auto dA_dX = grad_values_ptr[sample_i * dot_dimensions + i];
                        dA_dr += dX_dr * dA_dX;
                    }
                    positions_grad_ptr[global_atom_i * 3 + direction] += dA_dr;
                }
            }
        }
    }

    // ======================= gradient w.r.t. cell ========================= //
    auto cell_grad = torch::Tensor();
    if (all_cells.requires_grad()) {
        cell_grad = torch::zeros_like(all_cells);
        assert(cell_grad.is_contiguous() && cell_grad.is_cpu());
        auto cell_grad_ptr = cell_grad.data_ptr<double>();

        // find the index of the "structure" dimension in the samples
        auto sample_names = descriptor->sample_names();
        auto structure_dimension_it = std::find(
            std::begin(sample_names),
            std::end(sample_names),
            "structure"
        );
        if (structure_dimension_it == std::end(sample_names)) {
            C10_THROW_ERROR(ValueError,
                "could not find 'structure' in the samples, this calculator is missing it"
            );
        }
        int64_t structure_dimension = std::distance(std::begin(sample_names), structure_dimension_it);

        for (size_t block_i=0; block_i<n_blocks; block_i++) {
            auto block = descriptor->block_by_id(block_i);
            auto gradient = block->gradient("cell");

            auto structures = block->samples()->values().index({torch::indexing::Slice(), structure_dimension});

            auto samples = gradient->samples();
            auto samples_values = samples->values();
            auto samples_values_ptr = samples_values.data_ptr<int32_t>();
            assert(samples->names().size() == 1);
            assert(samples->names()[0] == "sample");

            // This is dX/ dH, which we computed in the forward pass
            auto forward_values = gradient->values();
            assert(forward_values.is_contiguous() && forward_values.is_cpu());
            auto forward_grad_ptr = forward_values.data_ptr<double>();

            // This is dA/ dX, which torch computed in the beginning of the
            // backward pass
            auto grad_output = grad_outputs[block_i];
            const auto& grad_output_sizes = grad_output.sizes();
            assert(grad_output.is_contiguous() && grad_output.is_cpu());
            auto grad_values_ptr = grad_output.data_ptr<double>();

            // total size of component + property dimension
            int64_t dot_dimensions = 1;
            for (int i=1; i<grad_output_sizes.size(); i++) {
                dot_dimensions *= grad_output_sizes[i];
            }

            // We want to compute dA/dH using the data above to finish the
            // backward pass
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = samples_values_ptr[grad_sample_i];
                // we get the structure from the samples of the values
                auto structure_i = structures[sample_i].item<int32_t>();

                for (int64_t direction_1=0; direction_1<3; direction_1++) {
                    for (int64_t direction_2=0; direction_2<3; direction_2++) {
                        auto dA_dH = 0.0;
                        for (int64_t i=0; i<dot_dimensions; i++) {
                            auto id = (grad_sample_i * 3 + direction_2) * 3 + direction_1;
                            auto dX_dH = forward_grad_ptr[id * dot_dimensions + i];
                            auto dA_dX = grad_values_ptr[sample_i * dot_dimensions + i];
                            dA_dH += dX_dH * dA_dX;
                        }
                        cell_grad_ptr[(structure_i * 3 + direction_1) * 3 + direction_2] += dA_dH;
                    }
                }
            }
        }
    }

    return {
        positions_grad,
        cell_grad,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
}
