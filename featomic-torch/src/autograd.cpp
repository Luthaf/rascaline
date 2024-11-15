#include <algorithm>

#include "metatensor/torch/tensor.hpp"
#include "featomic/torch/autograd.hpp"

#include "./openmp.hpp"

using namespace metatensor_torch;
using namespace featomic_torch;

// # NOTATION
//
// In this file, we are manipulating a lot of different gradients, so for easier
// reading here is the convention used:
//
// - A and B are things the user called `backward` on;
// - X is the representation that we are computing;
// - r is the atomic positions;
// - H is the cell matrix;

/// Implementation of the positions part of `FeatomicAutograd::backward` as
/// another custom autograd function, to allow for double backward.
template <typename scalar_t>
struct PositionsGrad: torch::autograd::Function<PositionsGrad<scalar_t>> {
    /// This operate one block at the time since we need to pass `dA_dX` (which
    /// comes from `FeatomicAutograd::backward` `grad_outputs`) as a
    /// `torch::Tensor` to be able to register a `grad_fn` with it.
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor all_positions,
        torch::Tensor dA_dX,
        TorchTensorBlock dX_dr,
        torch::IValue systems_start
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<torch::Tensor> grad_outputs
    );
};

/// Same as `PositionsGrad` but for cell gradients
template <typename scalar_t>
struct CellGrad: torch::autograd::Function<CellGrad<scalar_t>> {
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor all_cells,
        torch::Tensor dA_dX,
        TorchTensorBlock dX_dH,
        torch::Tensor systems
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<torch::Tensor> grad_outputs
    );
};


/******************************************************************************/
/*                            Helper functions                                */
/******************************************************************************/

#define stringify(str) #str
#define always_assert(condition)                                               \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error(                                          \
                std::string("assert failed ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + stringify(condition)         \
            );                                                                 \
        }                                                                      \
    } while (false)

static std::vector<TorchTensorBlock> extract_gradient_blocks(
    const TorchTensorMap& tensor,
    const std::string& parameter
) {
    auto gradients = std::vector<TorchTensorBlock>();
    for (int64_t i=0; i<tensor->keys()->count(); i++) {
        auto block = TensorMapHolder::block_by_id(tensor, i);
        auto gradient = TensorBlockHolder::gradient(block, parameter);

        gradients.push_back(torch::make_intrusive<TensorBlockHolder>(
            gradient->values(),
            gradient->samples(),
            gradient->components(),
            gradient->properties()
        ));
    }

    return gradients;
}

/******************************************************************************/
/*                           FeatomicAutograd                                */
/******************************************************************************/

std::vector<torch::Tensor> FeatomicAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor all_positions,
    torch::Tensor all_cells,
    torch::IValue systems_start,
    metatensor_torch::TorchTensorBlock block
) {
    ctx->save_for_backward({all_positions, all_cells});

    if (all_positions.requires_grad()) {
        ctx->saved_data.emplace("systems_start", systems_start);

        auto gradient = TensorBlockHolder::gradient(block, "positions");
        ctx->saved_data["positions_gradients"] = torch::make_intrusive<TensorBlockHolder>(
            gradient->values(),
            gradient->samples(),
            gradient->components(),
            gradient->properties()
        );
    }

    if (all_cells.requires_grad()) {
        ctx->saved_data["samples"] = block->samples();

        auto gradient = TensorBlockHolder::gradient(block, "cell");
        ctx->saved_data["cell_gradients"] = torch::make_intrusive<TensorBlockHolder>(
            gradient->values(),
            gradient->samples(),
            gradient->components(),
            gradient->properties()
        );
    }

    return {block->values()};
}

std::vector<torch::Tensor> FeatomicAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs
) {
    // ============== get the saved data from the forward pass ============== //
    auto saved_variables = ctx->get_saved_variables();
    auto all_positions = saved_variables[0];
    auto all_cells = saved_variables[1];

    always_assert(grad_outputs.size() == 1);
    // TODO: explore not not making this contiguous, instead using
    // torch::dot/torch::einsum if they are not too slow.
    grad_outputs[0] = grad_outputs[0].contiguous();

    auto positions_grad = torch::Tensor();
    auto cell_grad = torch::Tensor();

    // ===================== gradient w.r.t. positions ====================== //
    if (all_positions.requires_grad()) {
        auto forward_gradient = ctx->saved_data["positions_gradients"].toCustomClass<TensorBlockHolder>();
        auto systems_start = ctx->saved_data["systems_start"];

        if (all_positions.scalar_type() == torch::kFloat32) {
            auto output = PositionsGrad<float>::apply(
                all_positions,
                grad_outputs[0],
                forward_gradient,
                systems_start
            );

            positions_grad = output[0];
        } else if (all_positions.scalar_type() == torch::kFloat64) {
            auto output = PositionsGrad<double>::apply(
                all_positions,
                grad_outputs[0],
                forward_gradient,
                systems_start
            );

            positions_grad = output[0];
        } else {
            C10_THROW_ERROR(TypeError, "featomic only supports float64 and float32 data");
        }
    }

    // ======================= gradient w.r.t. cell ========================= //
    if (all_cells.requires_grad()) {
        auto forward_gradient = ctx->saved_data["cell_gradients"].toCustomClass<TensorBlockHolder>();
        auto block_samples = ctx->saved_data["samples"].toCustomClass<LabelsHolder>();

        // find the index of the "system" dimension in the samples
        const auto& sample_names = block_samples->names();
        auto system_dimension_it = std::find(
            std::begin(sample_names),
            std::end(sample_names),
            "system"
        );
        if (system_dimension_it == std::end(sample_names)) {
            C10_THROW_ERROR(ValueError,
                "could not find 'system' in the samples, this calculator is missing it"
            );
        }
        int64_t system_dimension = std::distance(std::begin(sample_names), system_dimension_it);

        auto systems = block_samples->values().index({torch::indexing::Slice(), system_dimension});

        if (all_cells.scalar_type() == torch::kFloat32) {
            auto output = CellGrad<float>::apply(
                all_cells,
                grad_outputs[0],
                forward_gradient,
                systems
            );

            cell_grad = output[0];
        } else if (all_cells.scalar_type() == torch::kFloat64) {
            auto output = CellGrad<double>::apply(
                all_cells,
                grad_outputs[0],
                forward_gradient,
                systems
            );

            cell_grad = output[0];
        } else {
            C10_THROW_ERROR(TypeError, "featomic only supports float64 and float32 data");
        }
    }

    return {
        positions_grad,
        cell_grad,
        torch::Tensor(),
        torch::Tensor(),
    };
}

/******************************************************************************/
/*                              PositionsGrad                                 */
/******************************************************************************/

template <typename scalar_t>
std::vector<torch::Tensor> PositionsGrad<scalar_t>::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor all_positions,
    torch::Tensor dA_dX,
    TorchTensorBlock dX_dr,
    torch::IValue systems_start_ivalue
) {
    // ====================== input parameters checks ======================= //
    always_assert(all_positions.requires_grad());
    auto systems_start = systems_start_ivalue.toIntList();

    auto samples = dX_dr->samples();
    const auto* sample_ptr = samples->as_metatensor().values().data();

    always_assert(samples->names().size() == 3);
    always_assert(samples->names()[0] == "sample");
    always_assert(samples->names()[1] == "system");
    always_assert(samples->names()[2] == "atom");

    // ========================= extract pointers =========================== //
    // TODO: remove all CPU <=> device data movement by rewriting the VJP
    // below with torch primitives
    auto dX_dr_values = dX_dr->values().to(torch::kCPU);
    always_assert(dX_dr_values.is_contiguous() && dX_dr_values.is_cpu());
    auto* dX_dr_ptr = dX_dr_values.data_ptr<scalar_t>();

    auto dA_dX_cpu = dA_dX.to(torch::kCPU);
    always_assert(dA_dX_cpu.is_contiguous() && dA_dX_cpu.is_cpu());
    auto* dA_dX_ptr = dA_dX_cpu.data_ptr<scalar_t>();

    // total size of component + property dimension
    const auto& dA_dX_sizes = dA_dX.sizes();
    int64_t n_features = 1;
    for (int i=1; i<dA_dX_sizes.size(); i++) {
        n_features *= dA_dX_sizes[i];
    }

    // =========================== compute dA_dr ============================ //
    // For OpenMP parallelization, we allocate a temporary output on each thread
    // with ThreadLocalTensor, then let each thread write to their own copy &
    // finally sum each of the thread local results.
    auto dA_dr_multiple = ThreadLocalTensor();
    #pragma omp parallel
    {
        #pragma omp single
        dA_dr_multiple.init(omp_get_num_threads(), all_positions.sizes(), all_positions.options());

        auto dA_dr_local = dA_dr_multiple.get();
        always_assert(dA_dr_local.is_contiguous() && dA_dr_local.is_cpu());
        auto dA_dr_ptr = dA_dr_local.data_ptr<scalar_t>();

        #pragma omp for
        for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
            auto sample_i = sample_ptr[grad_sample_i * 3 + 0];
            auto system_i = sample_ptr[grad_sample_i * 3 + 1];
            auto atom_i = sample_ptr[grad_sample_i * 3 + 2];

            auto global_atom_i = systems_start[system_i] + atom_i;

            for (int64_t xyz=0; xyz<3; xyz++) {
                auto dot = 0.0;
                for (int64_t i=0; i<n_features; i++) {
                    dot += (
                        dX_dr_ptr[(grad_sample_i * 3 + xyz) * n_features + i]
                        * dA_dX_ptr[sample_i * n_features + i]
                    );
                }
                dA_dr_ptr[global_atom_i * 3 + xyz] += dot;
            }
        }
    }
    auto dA_dr = dA_dr_multiple.sum();


    // ===================== data for double backward ======================= //
    ctx->save_for_backward({all_positions, dA_dX});
    ctx->saved_data.emplace("positions_gradients", dX_dr);
    ctx->saved_data.emplace("systems_start", systems_start_ivalue);

    return {dA_dr};
}

template <typename scalar_t>
std::vector<torch::Tensor> PositionsGrad<scalar_t>::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs
) {
    // ====================== input parameters checks ======================= //
    auto saved_variables = ctx->get_saved_variables();
    auto all_positions = saved_variables[0];
    auto dA_dX = saved_variables[1];

    auto dX_dr = ctx->saved_data["positions_gradients"].toCustomClass<TensorBlockHolder>();
    auto systems_start = ctx->saved_data["systems_start"].toIntList();

    auto dB_d_dA_dr = grad_outputs[0]; // gradient of B w.r.t. dA/dr (output of forward)

    auto samples = dX_dr->samples();
    const auto* sample_ptr = samples->as_metatensor().values().data();

    always_assert(samples->names().size() == 3);
    always_assert(samples->names()[0] == "sample");
    always_assert(samples->names()[1] == "system");
    always_assert(samples->names()[2] == "atom");

    // ========================= extract pointers =========================== //
    // TODO: remove all CPU <=> device data movement by rewriting the VJP
    // below with torch primitives
    auto dX_dr_values = dX_dr->values().to(torch::kCPU);
    always_assert(dX_dr_values.is_contiguous() && dX_dr_values.is_cpu());
    auto* dX_dr_ptr = dX_dr_values.data_ptr<scalar_t>();

    always_assert(dB_d_dA_dr.is_contiguous() && dB_d_dA_dr.is_cpu());
    auto* dB_d_dA_dr_ptr = dB_d_dA_dr.data_ptr<scalar_t>();


    auto dA_dX_cpu = dA_dX.to(torch::kCPU);
    always_assert(dA_dX_cpu.is_contiguous() && dA_dX_cpu.is_cpu());
    auto* dA_dX_ptr = dA_dX_cpu.data_ptr<scalar_t>();

    // total size of component + property dimension
    const auto& dA_dX_sizes = dA_dX.sizes();
    int64_t n_features = 1;
    for (int i=1; i<dA_dX_sizes.size(); i++) {
        n_features *= dA_dX_sizes[i];
    }

    // ================== gradient of B w.r.t. positions ==================== //
    auto dB_dr = torch::Tensor();
    if (all_positions.requires_grad()) {
        TORCH_WARN_ONCE(
            "second derivatives with respect to positions are not implemented "
            "and will not be accumulated during backward() calls. If you need "
            "second derivatives, please open an issue on featomic repository."
        );
    }

    // ============ gradient of B w.r.t. dA/dX (input of forward) =========== //
    auto dB_d_dA_dX = torch::Tensor();
    if (dA_dX.requires_grad()) {
        auto dB_d_dA_dX_multiple = ThreadLocalTensor();

        #pragma omp parallel
        {
            #pragma omp single
            dB_d_dA_dX_multiple.init(omp_get_num_threads(), dA_dX_cpu.sizes(), dA_dX_cpu.options());

            auto dB_d_dA_dX_local = dB_d_dA_dX_multiple.get();
            always_assert(dB_d_dA_dX_local.is_contiguous() && dB_d_dA_dX_local.is_cpu());
            auto* dB_d_dA_dX_ptr = dB_d_dA_dX_local.data_ptr<scalar_t>();

            // dX_dr.shape      == [positions gradient samples, 3, features...]
            // dB_d_dA_dr.shape == [n_atoms, 3]
            // dB_d_dA_dX.shape == [samples, features...]
            #pragma omp for
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = sample_ptr[3 * grad_sample_i + 0];
                auto system_i = sample_ptr[3 * grad_sample_i + 1];
                auto atom_i = sample_ptr[3 * grad_sample_i + 2];

                auto global_atom_i = systems_start[system_i] + atom_i;

                for (int64_t i=0; i<n_features; i++) {
                    auto dot = 0.0;
                    for (int64_t xyz=0; xyz<3; xyz++) {
                        dot += (
                            dX_dr_ptr[(grad_sample_i * 3 + xyz) * n_features + i]
                            * dB_d_dA_dr_ptr[global_atom_i * 3 + xyz]
                        );
                    }
                    dB_d_dA_dX_ptr[sample_i * n_features + i] += dot;
                }
            }
        }

        dB_d_dA_dX = dB_d_dA_dX_multiple.sum();
    }

    return {
        dB_dr,
        dB_d_dA_dX.to(dA_dX.device()),
        torch::Tensor(),
        torch::Tensor(),
    };
}


/******************************************************************************/
/*                                CellGrad                                    */
/******************************************************************************/

template <typename scalar_t>
std::vector<torch::Tensor> CellGrad<scalar_t>::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor all_cells,
    torch::Tensor dA_dX,
    TorchTensorBlock dX_dH,
    torch::Tensor systems
) {
    // ====================== input parameters checks ======================= //
    always_assert(all_cells.requires_grad());

    auto samples = dX_dH->samples();
    const auto* sample_ptr = samples->as_metatensor().values().data();

    always_assert(samples->names().size() == 1);
    always_assert(samples->names()[0] == "sample");

    // ========================= extract pointers =========================== //
    auto dX_dH_values = dX_dH->values().to(torch::kCPU);
    always_assert(dX_dH_values.is_contiguous() && dX_dH_values.is_cpu());
    auto* dX_dH_ptr = dX_dH_values.data_ptr<scalar_t>();

    // TODO: remove all CPU <=> device data movement by rewriting the VJP
    // below with torch primitives
    auto dA_dX_cpu = dA_dX.to(torch::kCPU);
    always_assert(dA_dX_cpu.is_contiguous() && dA_dX_cpu.is_cpu());
    auto* dA_dX_ptr = dA_dX_cpu.data_ptr<scalar_t>();

    const auto& dA_dX_sizes = dA_dX.sizes();
    // total size of component + property dimension
    int64_t n_features = 1;
    for (int i=1; i<dA_dX_sizes.size(); i++) {
        n_features *= dA_dX_sizes[i];
    }

    // =========================== compute dA_dH ============================ //
    auto dA_dH_multiple = ThreadLocalTensor();
    #pragma omp parallel
    {
        #pragma omp single
        dA_dH_multiple.init(omp_get_num_threads(), all_cells.sizes(), all_cells.options());

        auto dA_dH_local = dA_dH_multiple.get();
        always_assert(dA_dH_local.is_contiguous() && dA_dH_local.is_cpu());
        auto dA_dH_ptr = dA_dH_local.data_ptr<scalar_t>();

        #pragma omp for
        for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
            auto sample_i = sample_ptr[grad_sample_i];
            // we get the system index from the samples of the values
            auto system_i = static_cast<int64_t>(systems[sample_i].item<int32_t>());

            for (int64_t xyz_1=0; xyz_1<3; xyz_1++) {
                for (int64_t xyz_2=0; xyz_2<3; xyz_2++) {
                    auto dot = 0.0;
                    for (int64_t i=0; i<n_features; i++) {
                        auto sample_component_row = (grad_sample_i * 3 + xyz_1) * 3 + xyz_2;
                        dot += (
                            dA_dX_ptr[sample_i * n_features + i]
                            * dX_dH_ptr[sample_component_row * n_features + i]
                        );
                    }
                    dA_dH_ptr[(system_i * 3 + xyz_1) * 3 + xyz_2] += dot;
                }
            }
        }
    }
    auto dA_dH = dA_dH_multiple.sum();

    // ===================== data for double backward ======================= //
    ctx->save_for_backward({all_cells, dA_dX, systems});
    ctx->saved_data.emplace("cell_gradients", dX_dH);

    return {dA_dH};
}


template <typename scalar_t>
std::vector<torch::Tensor> CellGrad<scalar_t>::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs
) {
    // ====================== input parameters checks ======================= //
    auto saved_variables = ctx->get_saved_variables();
    auto all_cells = saved_variables[0];
    auto dA_dX = saved_variables[1];
    auto systems = saved_variables[2];

    auto dX_dH = ctx->saved_data["cell_gradients"].toCustomClass<TensorBlockHolder>();

    auto dB_d_dA_dH = grad_outputs[0]; // gradient of B w.r.t. dA/dH (output of forward)

    auto samples = dX_dH->samples();
    const auto* sample_ptr = samples->as_metatensor().values().data();
    always_assert(samples->names().size() == 1);
    always_assert(samples->names()[0] == "sample");

    // ========================= extract pointers =========================== //
    // TODO: remove all CPU <=> device data movement by rewriting the VJP
    // below with torch primitives
    auto dX_dH_values = dX_dH->values().to(torch::kCPU);
    always_assert(dX_dH_values.is_contiguous() && dX_dH_values.is_cpu());
    auto* dX_dH_ptr = dX_dH_values.data_ptr<scalar_t>();

    always_assert(dB_d_dA_dH.is_contiguous() && dB_d_dA_dH.is_cpu());
    auto* dB_d_dA_dH_ptr = dB_d_dA_dH.data_ptr<scalar_t>();

    auto dA_dX_cpu = dA_dX.to(torch::kCPU);
    always_assert(dA_dX_cpu.is_contiguous() && dA_dX_cpu.is_cpu());
    auto* dA_dX_ptr = dA_dX_cpu.data_ptr<scalar_t>();

    // total size of component + property dimension
    const auto& dA_dX_sizes = dA_dX.sizes();
    int64_t n_features = 1;
    for (int i=1; i<dA_dX_sizes.size(); i++) {
        n_features *= dA_dX_sizes[i];
    }

    // ===================== gradient of B w.r.t. cell ====================== //
    auto dB_dH = torch::Tensor();
    if (all_cells.requires_grad()) {
        TORCH_WARN_ONCE(
            "second derivatives with respect to cell matrix are not implemented "
            "and will not be accumulated during backward() calls. If you need "
            "second derivatives, please open an issue on featomic repository."
        );
    }

    // ============ gradient of B w.r.t. dA/dX (input of forward) =========== //
    auto dB_d_dA_dX = torch::Tensor();
    if (dA_dX.requires_grad()) {
        auto dB_d_dA_dX_multiple = ThreadLocalTensor();

        #pragma omp parallel
        {
            #pragma omp single
            dB_d_dA_dX_multiple.init(omp_get_num_threads(), dA_dX_cpu.sizes(), dA_dX_cpu.options());

            auto dB_d_dA_dX_local = dB_d_dA_dX_multiple.get();
            always_assert(dB_d_dA_dX_local.is_contiguous() && dB_d_dA_dX_local.is_cpu());
            auto* dB_d_dA_dX_ptr = dB_d_dA_dX_local.data_ptr<scalar_t>();

            // dX_dH.shape      == [cell gradient samples, 3, 3, features...]
            // dB_d_dA_dH.shape == [systems, 3, 3]
            // dB_d_dA_dX.shape == [samples, features...]
            #pragma omp for
            for (int64_t grad_sample_i=0; grad_sample_i<samples->count(); grad_sample_i++) {
                auto sample_i = sample_ptr[grad_sample_i];
                auto system_i = static_cast<int64_t>(systems[sample_i].item<int32_t>());

                for (int64_t i=0; i<n_features; i++) {
                    auto dot = 0.0;
                    for (int64_t xyz_1=0; xyz_1<3; xyz_1++) {
                        for (int64_t xyz_2=0; xyz_2<3; xyz_2++) {
                            auto idx_1 = (system_i * 3 + xyz_1) * 3 + xyz_2;
                            auto idx_2 = (grad_sample_i * 3 + xyz_1) * 3 + xyz_2;

                            dot += dB_d_dA_dH_ptr[idx_1] * dX_dH_ptr[idx_2 * n_features + i];
                        }
                    }
                    dB_d_dA_dX_ptr[sample_i * n_features + i] += dot;
                }
            }
        }

        dB_d_dA_dX = dB_d_dA_dX_multiple.sum();
    }

    return {
        dB_dH,
        dB_d_dA_dX.to(dA_dX.device()),
        torch::Tensor(),
        torch::Tensor(),
    };
}
