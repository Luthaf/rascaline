#ifndef RASCALINE_TORCH_CALCULATOR_HPP
#define RASCALINE_TORCH_CALCULATOR_HPP

#include <torch/script.h>

#include <rascaline.hpp>
#include <equistore/torch.hpp>

#include "rascaline/torch/exports.h"
#include "rascaline/torch/system.hpp"

namespace rascaline_torch {
class RascalineAutograd;

class CalculatorHolder;
using TorchCalculator = torch::intrusive_ptr<CalculatorHolder>;

/// Custom class holder to store, serialize and load rascaline calculators
/// inside Torch(Script) modules.
class RASCALINE_TORCH_EXPORT CalculatorHolder: public torch::CustomClassHolder {
public:
    /// Create a new calculator with the given `name` and JSON `parameters`
    CalculatorHolder(std::string name, std::string parameters):
        calculator_(std::move(name), std::move(parameters))
    {}

    /// Get the name of this calculator
    std::string name() const {
        return calculator_.name();
    }

    /// Get the parameters of this
    std::string parameters() const {
        return calculator_.parameters();
    }

    /// Run a calculation for the given `systems`.
    ///
    /// `gradients` controls which gradients will be stored in the
    /// output TensorMap
    equistore_torch::TorchTensorMap compute(
        std::vector<TorchSystem> systems
    );

private:
    friend class RascalineAutograd;

    /// Actual implementation of `compute`, used by `RascalineAutograd`
    equistore_torch::TorchTensorMap compute_impl(
        std::vector<TorchSystem>& systems,
        rascaline::CalculationOptions options
    );

    rascaline::Calculator calculator_;
};

}

#endif
