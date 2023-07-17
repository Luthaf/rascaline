#ifndef RASCALINE_TORCH_AUTOGRAD_HPP
#define RASCALINE_TORCH_AUTOGRAD_HPP

#include <torch/autograd.h>

#include <equistore/torch.hpp>

#include "rascaline/torch/exports.h"
#include "rascaline/torch/system.hpp"
#include "rascaline/torch/calculator.hpp"

namespace rascaline_torch {

/// Custom torch::autograd::Function integrating rascaline with torch autograd.
///
/// This is a bit more complex than your typical autograd because there is some
/// impedance mismatch between rascaline and torch. Most of it should be taken
/// care of by the `compute` function below.
class RASCALINE_TORCH_EXPORT RascalineAutograd: public torch::autograd::Function<RascalineAutograd> {
public:
    /// Compute the representation of the `systems` using the `calculator`,
    /// registering the operation as a node in Torch's computational graph.
    ///
    /// `_all_positions` and `all_cell` are only used to make sure torch
    /// registers nodes in the calculation graph. They must be the same as
    /// `torch::vstack([s->get_positions() for s in systems])` and
    /// `torch::vstack([s->get_cell() for s in systems])` respectively.
    ///
    /// This function "returns" an equistore TensorMap in it's last parameter,
    /// which should then be passed on to C++/Python code.
    ///
    /// This function also actually returns a list of torch::Tensor containing
    /// the values for each block in the TensorMap. This should be left unused,
    /// and is only there to make sure torch registers a `grad_fn` for the
    /// tensors stored inside the TensorMap (the tensors in the TensorMap are
    /// references to the ones returned by this function, so when a `grad_fn` is
    /// added to one, it is also added to the other).
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor all_positions,
        torch::Tensor all_cells,
        CalculatorHolder& calculator,
        std::vector<TorchSystem> systems,
        equistore_torch::TorchTensorMap* descriptor,
        std::vector<std::string> forward_gradients
    );

    /// Backward step: get the gradients of some quantity `A` w.r.t. the outputs
    /// of `forward`; and compute the gradients of the same quantity `A` w.r.t.
    /// the inputs of `forward` (i.e. cell and positions).
    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<torch::Tensor> grad_outputs
    );
};

}

#endif
