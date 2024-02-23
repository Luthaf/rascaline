#include <metatensor/torch.hpp>
#include <rascaline.hpp>

#include "rascaline/torch/calculator.hpp"
#include "rascaline/torch/autograd.hpp"
#include "rascaline/torch/system.hpp"

using namespace metatensor_torch;
using namespace rascaline_torch;

// move a block created by rascaline to torch
static TorchTensorBlock block_to_torch(
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
        sizes,
        [tensor](void*) mutable {
            // this function holds a copy of `tensor`, which will make sure that
            // (a) the TensorMap is kept alive for as long as the values
            // returned by `torch::from_blob` is; and (b) the TensorMap will be
            // freed once all the torch::Tensor created with `torch::from_blob`
            // are freed as well
            auto _ = std::move(tensor);
        },
        torch::TensorOptions().dtype(torch::kF64).device(torch::kCPU)
    );

    auto components = std::vector<TorchLabels>();
    components.reserve(block.components().size());
    for (auto component: block.components()) {
        components.emplace_back(torch::make_intrusive<LabelsHolder>(std::move(component)));
    }

    auto new_block = torch::make_intrusive<TensorBlockHolder>(
        torch_values,
        torch::make_intrusive<LabelsHolder>(block.samples()),
        std::move(components),
        torch::make_intrusive<LabelsHolder>(block.properties())
    );

    for (const auto& parameter: block.gradients_list()) {
        auto gradient = block_to_torch(tensor, block.gradient(parameter));
        new_block->add_gradient(parameter, std::move(gradient));
    }

    return new_block;
}

static torch::Tensor stack_all_positions(const std::vector<metatensor_torch::System>& systems) {
    auto all_positions = std::vector<torch::Tensor>();
    all_positions.reserve(systems.size());

    for (const auto& system: systems) {
        all_positions.push_back(system->positions().to(torch::kCPU));
    }

    return torch::vstack(all_positions);
}

static torch::Tensor stack_all_cells(const std::vector<metatensor_torch::System>& systems) {
    auto all_cells = std::vector<torch::Tensor>();
    all_cells.reserve(systems.size());

    for (const auto& system: systems) {
        all_cells.push_back(system->cell().to(torch::kCPU));
    }

    return torch::vstack(all_cells);
}

static bool all_systems_use_native(const std::vector<SystemAdapter>& systems) {
    auto result = systems[0].use_native_system();
    for (const auto& system: systems) {
        if (system.use_native_system() != result) {
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
        auto block = TensorMapHolder::block_by_id(tensor, i);
        auto new_block = torch::make_intrusive<TensorBlockHolder>(
            block->values(),
            block->samples(),
            block->components(),
            block->properties()
        );

        for (const auto& parameter: gradients_to_keep) {
            auto gradient = TensorBlockHolder::gradient(block, parameter);
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

static torch::ScalarType systems_dtype(const std::vector<metatensor_torch::System>& systems) {
    if (systems.empty()) {
        return torch::kFloat64;
    } else {
        auto dtype = systems[0]->scalar_type();
        for (const auto& system: systems) {
            if (system->scalar_type() != dtype) {
                C10_THROW_ERROR(TypeError,
                    std::string("all systems should have the same dtype, got ") +
                    torch::toString(system->scalar_type()) + " and " +
                    torch::toString(dtype)
                );
            }
        }
        return dtype;
    }
}

static torch::Device systems_device(const std::vector<metatensor_torch::System>& systems) {
    if (systems.empty()) {
        return torch::kCPU;
    } else {
        auto device = systems[0]->device();
        for (const auto& system: systems) {
            if (system->device() != device) {
                C10_THROW_ERROR(TypeError,
                    "all systems should have the same device, got " +
                    system->device().str() + " and " + device.str()
                );
            }
        }
        return device;
    }
}


metatensor_torch::TorchTensorMap CalculatorHolder::compute(
    std::vector<metatensor_torch::System> systems,
    TorchCalculatorOptions torch_options
) {
    auto dtype = systems_dtype(systems);
    auto device = systems_device(systems);

    if (!device.is_cpu()) {
        TORCH_WARN_ONCE(
            "Systems data is on device ", device, " but rascaline only supports ",
            "calculations on CPU. All the data will be moved to CPU and then "
            "back on device on your behalf"
        );
    }

    if (dtype != torch::kFloat32 && dtype != torch::kFloat64) {
        C10_THROW_ERROR(TypeError, "rascaline only supports float64 and float32 data");
    }

    auto all_positions = stack_all_positions(systems);
    auto all_cells = stack_all_cells(systems);
    auto systems_start_ivalue = torch::IValue();

    // =============== Handle all options for the calculation =============== //
    if (torch_options.get() == nullptr) {
        torch_options = torch::make_intrusive<CalculatorOptionsHolder>();
    }
    auto options = rascaline::CalculationOptions();

    // which gradients should we compute? We have to compute some gradient
    // either if positions/cell has `requires_grad` set to `true`, or if the
    // user requested specific gradients in `forward_gradients`
    for (const auto& parameter: torch_options->gradients) {
        if (parameter != "positions" && parameter != "cell") {
            C10_THROW_ERROR(ValueError, "invalid gradients requested: " + parameter);
        }
    }

    if (contains(torch_options->gradients, "positions") || all_positions.requires_grad()) {
        options.gradients.push_back("positions");

        auto systems_start = c10::List<int64_t>();
        int64_t current_start = 0;
        for (auto& system: systems) {
            systems_start.push_back(current_start);
            current_start += static_cast<int64_t>(system->size());
        }
        systems_start_ivalue = torch::IValue(std::move(systems_start));
    }

    if (contains(torch_options->gradients, "cell") || all_cells.requires_grad()) {
        options.gradients.push_back("cell");
    }

    // where all computed gradients explicitly requested in forward_gradients?
    bool all_forward_gradients = true;
    for (const auto& parameter: options.gradients) {
        if (!contains(torch_options->gradients, parameter)) {
            all_forward_gradients = false;
        }
    }

    // convert the systems
    auto rascaline_systems = std::vector<SystemAdapter>();
    rascaline_systems.reserve(systems.size());
    for (auto& system: systems) {
        rascaline_systems.emplace_back(system);
    }

    options.use_native_system = all_systems_use_native(rascaline_systems);
    if (torch_options->selected_keys().isCustomClass()) {
        options.selected_keys = torch_options->selected_keys().toCustomClass<LabelsHolder>()->as_metatensor();
    }
    options.selected_samples = torch_options->selected_samples_rascaline();
    options.selected_properties = torch_options->selected_properties_rascaline();

    // ============ run the calculation and move data to torch ============== //
    auto raw_descriptor = std::make_shared<metatensor::TensorMap>(
        calculator_.compute(rascaline_systems, options)
    );

    // move all data to torch
    auto blocks = std::vector<TorchTensorBlock>();
    blocks.reserve(raw_descriptor->keys().count());
    for (size_t block_i=0; block_i<raw_descriptor->keys().count(); block_i++) {
        blocks.emplace_back(block_to_torch(raw_descriptor, raw_descriptor->block_by_id(block_i)));
    }

    auto torch_descriptor = torch::make_intrusive<metatensor_torch::TensorMapHolder>(
        torch::make_intrusive<LabelsHolder>(raw_descriptor->keys()),
        std::move(blocks)
    );

    if (!systems.empty()) {
        torch_descriptor = torch_descriptor->to(systems[0]->scalar_type(), systems[0]->device());
    }

    // ============ register the autograd nodes for each block ============== //
    for (int64_t block_i=0; block_i<torch_descriptor->keys()->count(); block_i++) {
        auto block = TensorMapHolder::block_by_id(torch_descriptor, block_i);
        // see `RascalineAutograd::forward` for an explanation of what's happening
        auto _ = RascalineAutograd::apply(
            all_positions,
            all_cells,
            systems_start_ivalue,
            block
        );
    }

    // ====================== handle forward gradients ====================== //
    if (all_forward_gradients) {
        return torch_descriptor;
    } else {
        return remove_other_gradients(torch_descriptor, torch_options->gradients);
    }
}


metatensor_torch::TorchTensorMap rascaline_torch::register_autograd(
    std::vector<metatensor_torch::System> systems,
    metatensor_torch::TorchTensorMap precomputed,
    std::vector<std::string> forward_gradients
) {
    if (precomputed->keys()->count() == 0) {
        return precomputed;
    }

    auto all_positions = stack_all_positions(systems);
    auto all_cells = stack_all_cells(systems);
    auto systems_start_ivalue = torch::IValue();

    auto precomputed_gradients = TensorMapHolder::block_by_id(precomputed, 0)->gradients_list();

    if (all_positions.requires_grad()) {
        if (!contains(precomputed_gradients, "positions")) {
            C10_THROW_ERROR(ValueError,
                "expected the precomputed TensorMap to contain gradients with "
                "respect to 'positions' since one of the system `requires_grad` "
                "for its positions"
            );
        }

        auto systems_start = c10::List<int64_t>();
        int64_t current_start = 0;
        for (auto& system: systems) {
            systems_start.push_back(current_start);
            current_start += static_cast<int64_t>(system->size());
        }
        systems_start_ivalue = torch::IValue(std::move(systems_start));
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
        auto block = TensorMapHolder::block_by_id(precomputed, block_i);
        auto _ = RascalineAutograd::apply(
            all_positions,
            all_cells,
            systems_start_ivalue,
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


// ========================================================================== //

/// Selected keys/samples/properties are passed as `torch::IValue`, this
/// function checks that the `torch::IValue` contains data of the right type.
/// None and Labels are always allowed, and if `tensormap_ok` is true,
/// `TensorMap` is also accepted (key selection does not accept `TensorMap`).
static void check_selection_type(
    const torch::IValue& selection,
    std::string option,
    bool tensormap_ok
) {
    if (selection.isNone()) {
        // all good
    } else if (selection.isCustomClass()) {
        // check if we have either a Labels or TensorMap
        try {
            selection.toCustomClass<metatensor_torch::LabelsHolder>();
        } catch (const c10::Error&) {
            if (tensormap_ok) {
                try {
                    selection.toCustomClass<metatensor_torch::TensorMapHolder>();
                } catch (const c10::Error&) {
                    C10_THROW_ERROR(TypeError,
                        "invalid type for `" + option + "`, expected None, Labels or TensorMap, got "
                        + selection.type()->str()
                    );
                }
            } else {
                C10_THROW_ERROR(TypeError,
                    "invalid type for `" + option + "`, expected None or Labels, got "
                    + selection.type()->str()
                );
            }
        }
        // all good
    } else {
        if (tensormap_ok) {
            C10_THROW_ERROR(TypeError,
                "invalid type for `" + option + "`, expected None, Labels or TensorMap, got "
                + selection.type()->str()
            );
        } else {
            C10_THROW_ERROR(TypeError,
                "invalid type for `" + option + "`, expected None or Labels, got "
                + selection.type()->str()
            );
        }
    }
}

static rascaline::LabelsSelection selection_to_rascaline(const torch::IValue& selection, std::string field) {
    if (selection.isNone()) {
        return rascaline::LabelsSelection::all();
    } else if (selection.isCustomClass()) {
        try {
            auto subset = selection.toCustomClass<metatensor_torch::LabelsHolder>();
            return rascaline::LabelsSelection::subset(subset->as_metatensor());
        } catch (const c10::Error&) {
            try {
                auto predefined = selection.toCustomClass<metatensor_torch::TensorMapHolder>();
                return rascaline::LabelsSelection::predefined(predefined->as_metatensor());
            } catch (const c10::Error&) {
                C10_THROW_ERROR(TypeError,
                    "internal error: invalid type for `" + field + "`, got "
                    + selection.type()->str()
                );
            }
        }
    } else {
        C10_THROW_ERROR(TypeError,
            "internal error: invalid type for `" + field + "`, got "
            + selection.type()->str()
        );
    }
}

void CalculatorOptionsHolder::set_selected_samples(torch::IValue selection) {
    check_selection_type(selection, "selected_samples", true);
    selected_samples_ = std::move(selection);
}

rascaline::LabelsSelection CalculatorOptionsHolder::selected_samples_rascaline() const {
    return selection_to_rascaline(selected_samples_, "selected_samples");
}

void CalculatorOptionsHolder::set_selected_properties(torch::IValue selection) {
    check_selection_type(selection, "selected_properties", true);
    selected_properties_ = std::move(selection);
}

rascaline::LabelsSelection CalculatorOptionsHolder::selected_properties_rascaline() const {
    return selection_to_rascaline(selected_properties_, "selected_properties");
}

void CalculatorOptionsHolder::set_selected_keys(torch::IValue selection) {
    check_selection_type(selection, "selected_keys", false);
    selected_keys_ = std::move(selection);
}
