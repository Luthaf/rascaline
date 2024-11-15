// IWYU pragma: private; include "featomic/torch.hpp"

#ifndef FEATOMIC_TORCH_AUTOGRAD_HPP
#define FEATOMIC_TORCH_AUTOGRAD_HPP

#include <ATen/core/ivalue.h>
#include <torch/autograd.h>

#include <metatensor/torch.hpp>

#include "featomic/torch/exports.h"

namespace featomic_torch {

/// Custom torch::autograd::Function integrating featomic with torch autograd.
///
/// This is a bit more complex than your typical autograd because there is some
/// impedance mismatch between featomic and torch. Most of it should be taken
/// care of by the `compute` function below.
class FEATOMIC_TORCH_EXPORT FeatomicAutograd: public torch::autograd::Function<FeatomicAutograd> {
public:
    /// Register a pseudo node in Torch's computational graph going from
    /// `all_positions` and `all_cell` to the values in `block`; using the
    /// pre-computed gradients in `block`.
    ///
    /// If `all_positions.requires_grad` is True, `block` must have a
    /// `"positions"` gradient; and `systems_start` should contain the index of
    /// the first atom of each system in `all_positions`.
    ///
    /// If `all_cells.requires_grad` is True, `block` must have a `"cell"`
    /// gradient, and the block samples must contain a `"stucture"` dimension.
    ///
    /// This function returns a vector with one element corresponding to
    /// `block.values`, which should be left unused. It is only there to make
    /// sure torch registers a `grad_fn` for the tensors stored inside the
    /// TensorBlock (the values in the TensorBlock are references to the ones
    /// returned by this function, so when a `grad_fn` is added to one, it is
    /// also added to the other).
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor all_positions,
        torch::Tensor all_cells,
        torch::IValue systems_start,
        metatensor_torch::TorchTensorBlock block
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
