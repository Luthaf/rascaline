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

    /// Get the parameters of this calculator
    std::string parameters() const {
        return calculator_.parameters();
    }

    /// Get all radial cutoffs used by this `Calculator`'s neighbors lists
    std::vector<double> cutoffs() const {
        return calculator_.cutoffs();
    }

    /// Run a calculation for the given `systems`.
    ///
    /// `gradients` controls which gradients will be stored in the
    /// output TensorMap
    equistore_torch::TorchTensorMap compute(
        std::vector<TorchSystem> systems,
        std::vector<std::string> gradients = {}
    );

private:
    rascaline::Calculator calculator_;
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
equistore_torch::TorchTensorMap RASCALINE_TORCH_EXPORT register_autograd(
    std::vector<TorchSystem> systems,
    equistore_torch::TorchTensorMap precomputed,
    std::vector<std::string> forward_gradients
);

}

#endif
