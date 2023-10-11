#ifndef RASCALINE_TORCH_SYSTEM_HPP
#define RASCALINE_TORCH_SYSTEM_HPP

#include <vector>
#include <map>

#include <torch/script.h>

#include <rascaline.hpp>

#include "rascaline/torch/exports.h"

namespace rascaline_torch {

class SystemHolder;
/// TorchScript will always manipulate `SystemHolder` through a `torch::intrusive_ptr`
using TorchSystem = torch::intrusive_ptr<SystemHolder>;

/// Implementation of `rascaline::System` using torch tensors as backing memory
/// for all the data.
///
/// This can either be used with the Rust neighbor list implementation; or a set
/// of pre-computed neighbor lists can be added to the system.
class RASCALINE_TORCH_EXPORT SystemHolder final: public rascaline::System, public torch::CustomClassHolder {
public:
    /// Construct a `TorchSystem` with the given tensors.
    ///
    /// @param species 1D integer tensor containing the atoms/particles species
    /// @param positions 2D tensor (the shape should be `len(species) x 3`)
    ///     containing the atoms/particles positions
    /// @param cell 3x3 tensor containing the bounding box for periodic boundary
    ///     conditions, or full of 0 if the system is non-periodic.
    ///
    SystemHolder(torch::Tensor species, torch::Tensor positions, torch::Tensor cell);

    /// SystemHolder can not be copy constructed
    SystemHolder(const SystemHolder&) = delete;
    /// SystemHolder can not be copy assigned
    SystemHolder& operator=(const SystemHolder&) = delete;

    /// SystemHolder can be move constructed
    SystemHolder(SystemHolder&&) = default;
    /// SystemHolder can be move assigned
    SystemHolder& operator=(SystemHolder&&) = default;

    ~SystemHolder() override = default;

    /*========================================================================*/
    /*            Functions to implement rascaline::System                    */
    /*========================================================================*/

    /// @private
    uintptr_t size() const override {
        return static_cast<uintptr_t>(species_.size(0));
    }

    /// @private
    const int32_t* species() const override {
        return species_.data_ptr<int32_t>();
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
    const std::vector<rascal_pair_t>& pairs() const override;

    /// @private
    const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const override;

    /*========================================================================*/
    /*                 Functions to re-use pre-computed pairs                 */
    /*========================================================================*/

    /// Should we copy data to rascaline internal data structure and compute the
    /// neighbor list there? This is set to `true` by default, or `false` if
    /// a neighbor list has been added with `set_precomputed_pairs`.
    bool use_native_system() const;

    /// Set the list of pre-computed pairs to `pairs` (following the convention
    /// required by `rascaline::System::pairs`), and store the `cutoff` used to
    /// compute the pairs.
    void set_precomputed_pairs(double cutoff, std::vector<rascal_pair_t> pairs);

    /*========================================================================*/
    /*                 Functions for the Python interface                     */
    /*========================================================================*/

    /// Get the species for this system
    torch::Tensor get_species() {
        return species_;
    }

    /// Get the positions for this system
    torch::Tensor get_positions() {
        return positions_;
    }

    /// Get the cell for this system
    torch::Tensor get_cell() {
        return cell_;
    }

    /// @private implementation of __len__ for TorchScript
    int64_t len() const {
        return species_.size(0);
    }

    /// @private implementation of __str__ for TorchScript
    std::string str() const;

    // TODO: convert from a Dict[str, TorchTensorMap] for the interface with LAMMPS
    // static TorchSystem from_metatensor_dict();

private:
    torch::Tensor species_;
    torch::Tensor positions_;
    torch::Tensor cell_;

    struct PrecomputedPairs {
        std::vector<rascal_pair_t> pairs_;
        std::vector<std::vector<rascal_pair_t>> pairs_by_center_;
    };

    // all precomputed pairs we know about
    std::map<double, PrecomputedPairs> precomputed_pairs_;
    // last custom requested by `compute_neighbors`
    double last_cutoff_ = -1.0;
};

}

#endif
