#ifndef FEATOMIC_TORCH_CALCULATOR_HPP
#define FEATOMIC_TORCH_CALCULATOR_HPP

#include <torch/script.h>

#include <featomic.hpp>
#include <metatensor/torch.hpp>
#include <metatensor/torch/atomistic.hpp>

#include "featomic/torch/exports.h"

namespace featomic_torch {
class FeatomicAutograd;

class CalculatorHolder;
using TorchCalculator = torch::intrusive_ptr<CalculatorHolder>;

class CalculatorOptionsHolder;
using TorchCalculatorOptions = torch::intrusive_ptr<CalculatorOptionsHolder>;

/// Options for a single calculation
class FEATOMIC_TORCH_EXPORT CalculatorOptionsHolder: public torch::CustomClassHolder {
public:
    /// get the current selected samples
    torch::IValue selected_samples() const {
        return selected_samples_;
    }
    /// Set the selected samples to `selection`.
    ///
    /// The `IValue` can be `None` (no selection), an instance of
    /// `metatensor_torch::TorchLabels`, or an instance of
    /// `metatensor_torch::TorchTensorMap`.
    void set_selected_samples(torch::IValue selection);

    /// Get the selected samples in the format used by featomic
    featomic::LabelsSelection selected_samples_featomic() const;

    /// get the current selected properties
    torch::IValue selected_properties() const {
        return selected_properties_;
    }
    /// Set the selected properties to `selection`.
    ///
    /// The `IValue` can be `None` (no selection), an instance of
    /// `metatensor_torch::TorchLabels`, or an instance of
    /// `metatensor_torch::TorchTensorMap`.
    void set_selected_properties(torch::IValue selection);

    /// Get the selected properties in the format used by featomic
    featomic::LabelsSelection selected_properties_featomic() const;

    /// get the current selected keys
    torch::IValue selected_keys() const {
        return selected_keys_;
    }

    /// Set the selected properties to `selection`.
    ///
    /// The `IValue` can be `None` (no selection), or an instance of
    /// `metatensor_torch::TorchLabels`.
    void set_selected_keys(torch::IValue selection);

    /// which gradients to keep in the output of a calculation
    std::vector<std::string> gradients = {};

private:
    torch::IValue selected_samples_ = torch::IValue();
    torch::IValue selected_properties_ = torch::IValue();
    torch::IValue selected_keys_ = torch::IValue();
};

/// Custom class holder to store, serialize and load featomic calculators
/// inside Torch(Script) modules.
class FEATOMIC_TORCH_EXPORT CalculatorHolder: public torch::CustomClassHolder {
public:
    /// Create a new calculator with the given `name` and JSON `parameters`
    CalculatorHolder(std::string name, std::string parameters):
        c_name_(std::move(name)),
        calculator_(c_name_, std::move(parameters))
    {}

    /// Get the name of this calculator
    std::string name() const {
        return calculator_.name();
    }

    /// Get the name used to register this calculator
    std::string c_name() const {
        return c_name_;
    }

    /// Get the parameters of this calculator
    std::string parameters() const {
        return calculator_.parameters();
    }

    /// Get all radial cutoffs used by this `Calculator`'s neighbors lists
    std::vector<double> cutoffs() const {
        return calculator_.cutoffs();
    }

    /// Run a calculation for the given `systems` using the given options
    metatensor_torch::TorchTensorMap compute(
        std::vector<metatensor_torch::System> systems,
        TorchCalculatorOptions options = {}
    );

private:
    std::string c_name_;
    featomic::Calculator calculator_;
};


/// Register autograd nodes between `system.positions` and `system.cell` for
/// each of the systems and the values in the `precomputed` TensorMap.
///
/// This is an advanced function must users should not need to use.
///
/// The autograd nodes `backward()` function will use the gradients already
/// stored in `precomputed`, meaning that if any of the system's positions
/// `requires_grad`, `precomputed` must contain `"positions"` gradients.
/// Similarly, if any of the system's cell `requires_grad`, `precomputed` must
/// contain `"cell"` gradients.
///
/// `forward_gradients` controls which gradients are left inside the TensorMap.
metatensor_torch::TorchTensorMap FEATOMIC_TORCH_EXPORT register_autograd(
    std::vector<metatensor_torch::System> systems,
    metatensor_torch::TorchTensorMap precomputed,
    std::vector<std::string> forward_gradients
);

}

#endif
