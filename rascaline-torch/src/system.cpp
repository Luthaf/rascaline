#include <cassert>
#include <sstream>

#include "rascaline/torch/system.hpp"

using namespace rascaline_torch;

SystemHolder::SystemHolder(torch::Tensor species, torch::Tensor positions, torch::Tensor cell) {
    auto species_sizes = species.sizes();
    if (species_sizes.size() != 1) {
        C10_THROW_ERROR(ValueError,
            "atomic species tensor must be a 1D tensor"
        );
    }
    auto n_atoms = species_sizes[0];

    this->species_ = species.to(torch::kInt32).to(torch::kCPU).contiguous();
    if (this->species_.requires_grad()) {
        C10_THROW_ERROR(ValueError,
            "species can not have requires_grad=True"
        );
    }

    /**************************************************************************/
    auto positions_sizes = positions.sizes();
    if (positions_sizes.size() != 2 || positions_sizes[0] != n_atoms || positions_sizes[1] != 3) {
        C10_THROW_ERROR(ValueError,
            "the positions tensor must be a (n_atoms x 3) tensor"
        );
    }

    this->positions_ = positions.to(torch::kDouble).to(torch::kCPU).contiguous();

    /**************************************************************************/
    auto cell_sizes = cell.sizes();
    if (cell_sizes.size() != 2 || cell_sizes[0] != 3 || cell_sizes[1] != 3) {
        C10_THROW_ERROR(ValueError,
            "the cell tensor must be a (3 x 3) tensor"
        );
    }

    this->cell_ = cell.to(torch::kDouble).to(torch::kCPU).contiguous();
}

void SystemHolder::set_precomputed_pairs(double cutoff, std::vector<rascal_pair_t> pairs) {
    auto pairs_by_center = std::vector<std::vector<rascal_pair_t>>();
    pairs_by_center.resize(this->size());

    for (const auto& pair: pairs) {
        pairs_by_center[pair.first].push_back(pair);
        pairs_by_center[pair.second].push_back(pair);
    }

    precomputed_pairs_.emplace(
        cutoff,
        PrecomputedPairs{std::move(pairs), std::move(pairs_by_center)}
    );
}

bool SystemHolder::use_native_system() const {
    return precomputed_pairs_.empty();
}

void SystemHolder::compute_neighbors(double cutoff) {
    if (this->use_native_system()) {
        C10_THROW_ERROR(ValueError,
            "this system only support 'use_native_systems=true'"
        );
    }

    if (cutoff <= 0.0) {
        C10_THROW_ERROR(ValueError,
            "cutoff can not be negative in `compute_neighbors`"
        );
    }

    // check that the pairs for this cutoff were already added with
    // `set_precomputed_pairs`
    if (precomputed_pairs_.find(cutoff) == std::end(precomputed_pairs_)) {
        auto message = std::ostringstream();
        message << "trying to get neighbor list with a cutoff (";
        message << cutoff << ") for which no pre-computed neighbor lists has been registered";
        message << " (we have lists for cutoff=[";

        int entry_i = 0;
        for (const auto& entry: precomputed_pairs_) {
            message << entry.first;

            // don't add a comma after the last element
            if (entry_i < precomputed_pairs_.size() - 1) {
                message << ", ";
            }
            entry_i += 1;
        }
        message << "])";

        C10_THROW_ERROR(ValueError, message.str());
    }

    last_cutoff_ = cutoff;
}

const std::vector<rascal_pair_t>& SystemHolder::pairs() const {
    if (this->use_native_system() || last_cutoff_ == -1.0) {
        C10_THROW_ERROR(ValueError,
            "this system only support 'use_native_systems=true'"
        );
    }

    auto it = precomputed_pairs_.find(last_cutoff_);
    assert(it != std::end(precomputed_pairs_));
    return it->second.pairs_;
}

const std::vector<rascal_pair_t>& SystemHolder::pairs_containing(uintptr_t center) const {
    if (this->use_native_system() || last_cutoff_ == -1.0) {
        C10_THROW_ERROR(ValueError,
            "this system only support 'use_native_systems=true'"
        );
    }

    auto it = precomputed_pairs_.find(last_cutoff_);
    assert(it != std::end(precomputed_pairs_));
    return it->second.pairs_by_center_[center];
}


std::string SystemHolder::__str__() const {
    auto result = std::ostringstream();
    result << "System with " << this->size() << " atoms, ";
    if (torch::all(cell_ == torch::zeros({3, 3})).item<bool>()) {
        result << "non periodic";
    } else {
        result << "periodic cell: [";
        for (int64_t i=0; i<3; i++) {
            for (int64_t j=0; j<3; j++) {
                result << cell_.index({i, j}).item<double>();
                if (j != 2 || i != 2) {
                    result << ", ";
                }
            }
        }
        result << "]";
    }

    return result.str();
}
