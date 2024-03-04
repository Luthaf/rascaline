#include <cassert>
#include <sstream>

#include "rascaline/torch/system.hpp"

using namespace rascaline_torch;

SystemAdapter::SystemAdapter(metatensor_torch::System system): system_(std::move(system)) {
    this->types_ = system_->types().to(torch::kCPU).contiguous();
    this->positions_ = system_->positions().to(torch::kCPU).to(torch::kDouble).contiguous();
    this->cell_ = system_->cell().to(torch::kCPU).to(torch::kDouble).contiguous();

    // convert all neighbors list that where requested by rascaline
    for (const auto& options: system_->known_neighbors_lists()) {
        for (const auto& requestor: options->requestors()) {
            if (requestor == "rascaline") {
                auto neighbors = system->get_neighbors_list(options);
                auto samples_values = neighbors->samples()->values().to(torch::kCPU).contiguous();
                auto samples = samples_values.accessor<int32_t, 2>();

                auto distances_tensor = neighbors->values().reshape({-1, 3}).to(torch::kCPU).to(torch::kDouble).contiguous();
                auto distances = distances_tensor.accessor<double, 2>();

                auto n_pairs = samples.size(1);

                auto pairs = std::vector<rascal_pair_t>();
                pairs.reserve(static_cast<size_t>(n_pairs));
                for (int64_t i=0; i<n_pairs; i++) {
                    auto x = distances[i][0];
                    auto y = distances[i][1];
                    auto z = distances[i][2];

                    auto pair = rascal_pair_t {};
                    pair.first = static_cast<uintptr_t>(samples[i][0]);
                    pair.second = static_cast<uintptr_t>(samples[i][1]);

                    pair.distance = std::sqrt(x*x + y*y + z*z);
                    pair.vector[0] = x;
                    pair.vector[1] = y;
                    pair.vector[2] = z;

                    pair.cell_shift_indices[0] = samples[i][2];
                    pair.cell_shift_indices[1] = samples[i][3];
                    pair.cell_shift_indices[2] = samples[i][4];

                    pairs.emplace_back(pair);
                }

                this->set_precomputed_pairs(options->cutoff(), std::move(pairs));
                continue;
            }
        }
    }
}

void SystemAdapter::set_precomputed_pairs(double cutoff, std::vector<rascal_pair_t> pairs) {
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

bool SystemAdapter::use_native_system() const {
    return precomputed_pairs_.empty();
}

void SystemAdapter::compute_neighbors(double cutoff) {
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

const std::vector<rascal_pair_t>& SystemAdapter::pairs() const {
    if (this->use_native_system() || last_cutoff_ == -1.0) {
        C10_THROW_ERROR(ValueError,
            "this system only support 'use_native_systems=true'"
        );
    }

    auto it = precomputed_pairs_.find(last_cutoff_);
    assert(it != std::end(precomputed_pairs_));
    return it->second.pairs_;
}

const std::vector<rascal_pair_t>& SystemAdapter::pairs_containing(uintptr_t atom) const {
    if (this->use_native_system() || last_cutoff_ == -1.0) {
        C10_THROW_ERROR(ValueError,
            "this system only support 'use_native_systems=true'"
        );
    }

    auto it = precomputed_pairs_.find(last_cutoff_);
    assert(it != std::end(precomputed_pairs_));
    return it->second.pairs_by_atom_[atom];
}
