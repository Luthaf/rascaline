#ifndef FEATOMIC_TORCH_SYSTEM_HPP
#define FEATOMIC_TORCH_SYSTEM_HPP

#include <vector>
#include <map>

#include <torch/script.h>

#include <featomic.hpp>
#include <metatensor/torch/atomistic.hpp>

#include "featomic/torch/exports.h"

namespace featomic_torch {

/// Implementation of `featomic::System` using `metatensor_torch::System` as
/// backing memory for all the data.
///
/// This can either be used with the Rust neighbor list implementation; or a set
/// of pre-computed neighbor lists can be added to the system.
class FEATOMIC_TORCH_EXPORT SystemAdapter final: public featomic::System {
public:
    /// Create a `SystemAdapter` wrapping an existing `metatensor_torch::System`
    SystemAdapter(metatensor_torch::System system);

    ~SystemAdapter() override = default;

    /// `SystemAdapter` is copy-constructible
    SystemAdapter(const SystemAdapter&) = default;
    /// `SystemAdapter` is move-constructible
    SystemAdapter(SystemAdapter&&) = default;

    /// `SystemAdapter` can be copy-assigned
    SystemAdapter& operator=(const SystemAdapter&) = default;
    /// `SystemAdapter` can be move-assigned
    SystemAdapter& operator=(SystemAdapter&&) = default;

    /*========================================================================*/
    /*            Functions to implement featomic::System                    */
    /*========================================================================*/

    /// @private
    uintptr_t size() const override {
        return static_cast<uintptr_t>(types_.size(0));
    }

    /// @private
    const int32_t* types() const override {
        return types_.data_ptr<int32_t>();
    }

    /// @private
    const double* positions() const override {
        return positions_.data_ptr<double>();
    }

    /// @private
    CellMatrix cell() const override {
        auto* data = cell_.data_ptr<double>();
        return CellMatrix{{
            {{data[0], data[1], data[2]}},
            {{data[3], data[4], data[5]}},
            {{data[6], data[7], data[8]}},
        }};
    }

    /// @private
    void compute_neighbors(double cutoff) override;

    /// @private
    const std::vector<featomic_pair_t>& pairs() const override;

    /// @private
    const std::vector<featomic_pair_t>& pairs_containing(uintptr_t atom) const override;

    /*========================================================================*/
    /*                 Functions to re-use pre-computed pairs                 */
    /*========================================================================*/

    /// Should we copy data to featomic internal data structure and compute the
    /// neighbor list there? This is set to `true` by default, or `false` if
    /// a neighbor list has been added with `set_precomputed_pairs`.
    bool use_native_system() const;

private:
    // the origin of all the data
    metatensor_torch::System system_;

    /// atomic types tensor, contiguous and on CPU
    torch::Tensor types_;
    /// positions tensor, contiguous, on CPU and with dtype=float64
    torch::Tensor positions_;
    /// cell tensor, contiguous, on CPU and with dtype=float64
    torch::Tensor cell_;


    struct PrecomputedPairs {
        std::vector<featomic_pair_t> pairs_;
        std::vector<std::vector<featomic_pair_t>> pairs_by_atom_;
    };

    void set_precomputed_pairs(double cutoff, std::vector<featomic_pair_t> pairs);

    // all precomputed pairs we know about
    std::map<double, PrecomputedPairs> precomputed_pairs_;
    // last custom requested by `compute_neighbors`
    double last_cutoff_ = -1.0;
};

}

#endif
