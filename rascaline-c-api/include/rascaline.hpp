#ifndef RASCALINE_HPP
#define RASCALINE_HPP

#include <cassert>
#include <cstring>
#include <cstdint>

#include <array>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <utility>
#include <stdexcept>
#include <exception>
#include <unordered_map>

#include "equistore.h"
#include "equistore.hpp"
#include "rascaline.h"

/// This file contains the C++ API to rascaline, manually built on top of the C
/// API defined in `rascaline.h`. This API uses the standard C++ library where
/// convenient, but also allow to drop back to the C API if required, by
/// providing functions to extract the C API handles (named `as_rascal_XXX`).

namespace rascaline {

/// Exception class for all error thrown by rascaline
class RascalineError : public std::runtime_error {
public:
    /// Create a new error with the given message
    RascalineError(std::string message): std::runtime_error(message) {}
    ~RascalineError() = default;

    /// RascalineError is copy-constructible
    RascalineError(const RascalineError&) = default;
    /// RascalineError is move-constructible
    RascalineError(RascalineError&&) = default;
    /// RascalineError can be copy-assigned
    RascalineError& operator=(const RascalineError&) = default;
    /// RascalineError can be move-assigned
    RascalineError& operator=(RascalineError&&) = default;
};

namespace details {
    /// Class able to store exceptions and retrieve them later
    class ExceptionsStore {
    public:
        ExceptionsStore(): map_(), next_id_(-1) {}

        ExceptionsStore(const ExceptionsStore&) = delete;
        ExceptionsStore(ExceptionsStore&&) = delete;
        ExceptionsStore& operator=(const ExceptionsStore&) = delete;
        ExceptionsStore& operator=(ExceptionsStore&&) = delete;

        /// Save an exception pointer inside the exceptions store and return the
        /// corresponding id as a **negative** integer.
        int32_t save_exception(std::exception_ptr exception) {
            auto id = next_id_;

            // this should not underflow, but better safe than sorry
            if (next_id_ == INT32_MIN) {
                throw RascalineError("too many exceptions, what are you doing???");
            }
            next_id_ -= 1;

            map_.emplace(id, std::move(exception));
            return id;
        }

        /// Get the exception pointer corresponding to the given exception id.
        /// The id **MUST** have been generated by a previous call to
        /// `save_exception`.
        std::exception_ptr extract_exception(int32_t id) {
            auto it = map_.find(id);
            if (it == map_.end()) {
                throw RascalineError("internal error: tried to access a non-existing exception");
            }

            auto exception = it->second;
            map_.erase(it);

            return exception;
        }

    private:
        std::unordered_map<int32_t, std::exception_ptr> map_;
        int32_t next_id_;
    };

    /// Singleton version of `ExceptionsStore`, protected by a mutex to be safe
    /// to call in multi-threaded context
    class GlobalExceptionsStore {
    public:
        /// Save an exception pointer inside the exceptions store and return the
        /// corresponding id as a **negative** integer.
        static int32_t save_exception(std::exception_ptr exception) {
            const std::lock_guard<std::mutex> lock(GlobalExceptionsStore::mutex());
            auto& store = GlobalExceptionsStore::instance();
            return store.save_exception(std::move(exception));
        }

        /// Get the exception pointer corresponding to the given exception id.
        /// The id **MUST** have been generated by a previous call to
        /// `save_exception`.
        static std::exception_ptr extract_exception(int32_t id) {
            const std::lock_guard<std::mutex> lock(GlobalExceptionsStore::mutex());
            auto& store = GlobalExceptionsStore::instance();
            return store.extract_exception(id);
        }

    private:
        /// the actual instance of the store, as a static singleton
        static ExceptionsStore& instance() {
            static ExceptionsStore instance;
            return instance;
        }

        /// mutex used to lock the map in multi-threaded context
        static std::mutex& mutex() {
            static std::mutex mutex;
            return mutex;
        }
    };

    /// Check the status returned by a rascal function, throwing an exception
    /// with the latest error message if the status is not `RASCAL_SUCCESS`.
    inline void check_status(rascal_status_t status) {
        if (status > RASCAL_SUCCESS) {
            throw RascalineError(rascal_last_error());
        } else if (status < RASCAL_SUCCESS) {
            // this error comes from C++, let's restore it and pass it up
            auto exception = GlobalExceptionsStore::extract_exception(status);
            std::rethrow_exception(exception);
        }
    }
}

#define RASCAL_SYSTEM_CATCH_EXCEPTIONS(__code__)                                \
    do {                                                                        \
        try {                                                                   \
            __code__                                                            \
            return RASCAL_SUCCESS;                                              \
        } catch (...) {                                                         \
            auto e = std::current_exception();                                  \
            return details::GlobalExceptionsStore::save_exception(std::move(e));\
        }                                                                       \
    } while (false)

/// A `System` deals with the storage of atoms and related information, as well
/// as the computation of neighbor lists.
///
/// This class only defines a pure virtual interface for `System`. In order to
/// provide access to new system, users must create a child class implementing
/// all virtual member functions.
class System {
public:
    System() = default;
    virtual ~System() = default;

    /// System is copy-constructible
    System(const System&) = default;
    /// System is move-constructible
    System(System&&) = default;
    /// System can be copy-assigned
    System& operator=(const System&) = default;
    /// System can be move-assigned
    System& operator=(System&&) = default;

    /// Get the number of atoms in this system
    virtual uintptr_t size() const = 0;

    /// Get a pointer to the first element a contiguous array (typically
    /// `std::vector` or memory allocated with `new[]`) containing the atomic
    /// species of each atom in this system. Different atomics species should be
    /// identified with a different value. These values are usually the atomic
    /// number, but don't have to be. The array should contain `System::size()`
    /// elements.
    virtual const int32_t* species() const = 0;

    /// Get a pointer to the first element of a contiguous array containing the
    /// atomic cartesian coordinates. `positions[0], positions[1], positions[2]`
    /// must contain the x, y, z cartesian coordinates of the first atom, and so
    /// on. The array should contain `3 x System::size()` elements.
    virtual const double* positions() const = 0;

    /// Unit cell representation as a 3x3 matrix. The cell should be written in
    /// row major order, i.e. `{{ax ay az}, {bx by bz}, {cx cy cz}}`, where
    /// a/b/c are the unit cell vectors.
    using CellMatrix = std::array<std::array<double, 3>, 3>;

    /// Get the matrix describing the unit cell
    virtual CellMatrix cell() const = 0;

    /// Compute the neighbor list with the given `cutoff`, and store it for
    /// later access using `System::pairs` or `System::pairs_containing`.
    virtual void compute_neighbors(double cutoff) = 0;

    /// Get the list of pairs in this system
    ///
    /// This list of pair should only contain each pair once (and not twice as
    /// `i-j` and `j-i`), should not contain self pairs (`i-i`); and should only
    /// contains pairs where the distance between atoms is actually bellow the
    /// cutoff passed in the last call to `System::compute_neighbors`. This
    /// function is only valid to call after a call to
    /// `System::compute_neighbors`.
    virtual const std::vector<rascal_pair_t>& pairs() const = 0;

    /// Get the list of pairs in this system containing the atom with index
    /// `center`.
    ///
    /// The same restrictions on the list of pairs as `System::pairs` applies,
    /// with the additional condition that the pair `i-j` should be included
    /// both in the return of `System::pairs_containing(i)` and
    /// `System::pairs_containing(j)`.
    virtual const std::vector<rascal_pair_t>& pairs_containing(uintptr_t center) const = 0;

    /// Convert a child instance of the `System` class to a `rascal_system_t` to
    /// be passed to the rascaline functions.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    rascal_system_t as_rascal_system_t() {
        return rascal_system_t {
            // user_data
            static_cast<void*>(this),
            // size
            [](const void* self, uintptr_t* size) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    *size = static_cast<const System*>(self)->size();
                );
            },
            // species
            [](const void* self, const int32_t** species) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    *species = static_cast<const System*>(self)->species();
                );
            },
            // positions
            [](const void* self, const double** positions) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    *positions = (reinterpret_cast<const System*>(self))->positions();
                );
            },
            // cell
            [](const void* self, double* cell) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    auto cpp_cell = reinterpret_cast<const System*>(self)->cell();
                    std::memcpy(cell, &cpp_cell[0][0], 9 * sizeof(double));
                );
            },
            // compute_neighbors
            [](void* self, double cutoff) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    reinterpret_cast<System*>(self)->compute_neighbors(cutoff);
                );
            },
            // pairs
            [](const void* self, const rascal_pair_t** pairs, uintptr_t* size) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    const auto& cpp_pairs = reinterpret_cast<const System*>(self)->pairs();
                    *pairs = cpp_pairs.data();
                    *size = cpp_pairs.size();
                );
            },
            // pairs_containing
            [](const void* self, uintptr_t center, const rascal_pair_t** pairs, uintptr_t* size) {
                RASCAL_SYSTEM_CATCH_EXCEPTIONS(
                    const auto& cpp_pairs = reinterpret_cast<const System*>(self)->pairs_containing(center);
                    *pairs = cpp_pairs.data();
                    *size = cpp_pairs.size();
                );
            }
        };
    }
};

#undef RASCAL_SYSTEM_CATCH_EXCEPTIONS


/// A collection of systems read from a file. This is a convenience class
/// enabling the common use case of reading systems from a file and runnning a
/// calculation on the systems. If you need more control or access to advanced
/// functionalities, you should consider writing a new class extending `System`.
class BasicSystems {
public:
    /// Read all structures in the file at the given `path` using
    /// [chemfiles](https://chemfiles.org/).
    ///
    /// This function can read all [formats supported by
    /// chemfiles](https://chemfiles.org/chemfiles/latest/formats.html).
    ///
    /// @throws RascalineError if chemfiles can not read the file
    BasicSystems(std::string path): systems_(nullptr), count_(0) {
        details::check_status(rascal_basic_systems_read(path.c_str(), &systems_, &count_));
    }

    ~BasicSystems() {
        details::check_status(rascal_basic_systems_free(systems_, count_));
    }

    /// BasicSystems is **NOT** copy-constructible
    BasicSystems(const BasicSystems&) = delete;
    /// BasicSystems can **NOT** be copy-assigned
    BasicSystems& operator=(const BasicSystems&) = delete;

    /// BasicSystems is move-constructible
    BasicSystems(BasicSystems&& other) {
        *this = std::move(other);
    }

    /// BasicSystems can be move-assigned
    BasicSystems& operator=(BasicSystems&& other) {
        this->~BasicSystems();
        this->systems_ = nullptr;
        this->count_ = 0;

        std::swap(this->systems_, other.systems_);
        std::swap(this->count_, other.count_);

        return *this;
    }

    /// Get a pointer to the first element of the underlying array of systems
    ///
    /// This function is intended for internal use only.
    rascal_system_t* systems() {
        return systems_;
    }

    /// Get the number of systems managed by this `BasicSystems`
    uintptr_t count() const {
        return count_;
    }

private:
    rascal_system_t* systems_ = nullptr;
    uintptr_t count_ = 0;
};

/// Rules to select labels (either samples or properties) on which the user
/// wants to run a calculation
class LabelsSelection {
public:
    /// Use all possible labels for this calculation
    static LabelsSelection all() {
        return LabelsSelection(nullptr, nullptr);
    }

    /// Select a subset of labels, using the same selection criterion for all
    /// keys in the final `TensorMap`.
    ///
    /// If the `selection` contains the same variables as the full set of
    /// labels, then only entries from the full set that also appear in this
    /// selection will be used.
    ///
    /// If the `selection` contains a subset of the variables of the full set of
    /// labels, then only entries from the full set which match one of the entry
    /// in this selection for all of the selection variable will be used.
    static LabelsSelection subset(std::shared_ptr<equistore::Labels> selection) {
        return LabelsSelection(std::move(selection), nullptr);
    }

    /// Use a predefined subset of labels, with different entries for different
    /// keys of the final `TensorMap`.
    ///
    /// For each key, the corresponding labels are fetched out of the
    /// `selection`, which must have the same set of keys as the full
    /// calculation.
    static LabelsSelection predefined(std::shared_ptr<equistore::TensorMap> selection) {
        return LabelsSelection(nullptr, std::move(selection));
    }

    ~LabelsSelection() = default;

    /// LabelsSelection can be copy-constructed
    LabelsSelection(const LabelsSelection& other): LabelsSelection(nullptr, nullptr) {
        *this = other;
    }

    /// LabelsSelection can be copy-assigned
    LabelsSelection& operator=(const LabelsSelection& other) {
        this->subset_ = other.subset_;
        this->predefined_ = other.predefined_;
        if (this->subset_ != nullptr) {
            this->c_subset_ = subset_->as_eqs_labels_t();
        }

        return *this;
    }

    /// LabelsSelection can be move-constructed
    LabelsSelection(LabelsSelection&& other): LabelsSelection(nullptr, nullptr) {
        *this = std::move(other);
    }

    /// LabelsSelection can be move-assigned
    LabelsSelection& operator=(LabelsSelection&& other) {
        this->subset_ = std::move(other.subset_);
        this->predefined_ = std::move(other.predefined_);
        if (this->subset_ != nullptr) {
            this->c_subset_ = subset_->as_eqs_labels_t();
        }

        other.subset_ = nullptr;
        other.predefined_ = nullptr;
        std::memset(&other.c_subset_, 0, sizeof(eqs_labels_t));

        return *this;
    }

    /// Get the `rascal_labels_selection_t` corresponding to this LabelsSelection
    rascal_labels_selection_t as_rascal_labels_selection_t() const {
        auto selection = rascal_labels_selection_t{};
        std::memset(&selection, 0, sizeof(rascal_labels_selection_t));

        if (subset_ != nullptr) {
            selection.subset = &c_subset_;
        }

        if (predefined_ != nullptr) {
            selection.predefined = predefined_->as_eqs_tensormap_t();
        }

        return selection;
    }

private:
    LabelsSelection(std::shared_ptr<equistore::Labels> subset, std::shared_ptr<equistore::TensorMap> predefined):
        subset_(std::move(subset)), c_subset_(), predefined_(std::move(predefined))
    {
        std::memset(&c_subset_, 0, sizeof(eqs_labels_t));
        if (subset_ != nullptr) {
            c_subset_ = subset_->as_eqs_labels_t();
        }
    }

    std::shared_ptr<equistore::Labels> subset_;
    eqs_labels_t c_subset_;
    std::shared_ptr<equistore::TensorMap> predefined_;
};


/// Options that can be set to change how a calculator operates.
class CalculationOptions {
public:
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    bool use_native_system = false;

    /// Selection of samples on which to run the computation
    LabelsSelection selected_samples = LabelsSelection::all();
    /// Selection of properties to compute for the samples
    LabelsSelection selected_properties = LabelsSelection::all();

    /// @verbatim embed:rst:leading-slashes
    /// List of gradients that should be computed. If this list is empty no
    /// gradients are computed.
    ///
    /// The following gradients are available:
    ///
    /// - ``"positions"``, for gradients of the representation with respect to
    ///   atomic positions. Positions gradients are computed as
    ///
    ///   .. math::
    ///       \frac{\partial \langle q \vert A_i \rangle}
    ///            {\partial \mathbf{r_j}}
    ///
    ///   where :math:`\langle q \vert A_i \rangle` is the representation around
    ///   atom :math:`i` and :math:`\mathbf{r_j}` is the position vector of the
    ///   atom :math:`j`.
    ///
    ///   **Note**: Position gradients of an atom are computed with respect to all
    ///   other atoms within the representation. To recover the force one has to
    ///   accumulate all pairs associated with atom :math:`i`.
    ///
    /// - ``"cell"``, for gradients of the representation with respect to cell
    ///   vectors. Cell gradients are computed as
    ///
    ///   .. math::
    ///       \frac{\partial \langle q \vert A_i \rangle}
    ///            {\partial \mathbf{h}}
    ///
    ///   where :math:`\mathbf{h}` is the cell matrix.
    ///
    ///   **Note**: When computing the virial, one often needs to evaluate
    ///   the gradient of the representation with respect to the strain
    ///   :math:`\epsilon`. To recover the typical expression from the cell
    ///   gradient one has to multiply the cell gradients with the cell
    ///   matrix :math:`\mathbf{h}`
    ///
    ///   .. math::
    ///       -\frac{\partial \langle q \vert A \rangle}
    ///             {\partial\epsilon}
    ///         = -\frac{\partial \langle q \vert A \rangle}
    ///                 {\partial \mathbf{h}} \cdot \mathbf{h}
    /// @endverbatim
    std::vector<const char*> gradients;

    /// Convert this instance of `CalculationOptions` to a
    /// `rascal_calculation_options_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    rascal_calculation_options_t as_rascal_calculation_options_t() const {
        auto options = rascal_calculation_options_t{};
        std::memset(&options, 0, sizeof(rascal_calculation_options_t));

        options.use_native_system = this->use_native_system;

        options.gradients = this->gradients.data();
        options.gradients_count = this->gradients.size();

        options.selected_samples = this->selected_samples.as_rascal_labels_selection_t();
        options.selected_properties = this->selected_properties.as_rascal_labels_selection_t();

        return options;
    }
};


/// The `Calculator` class implements the calculation of a given atomic scale
/// representation. Specific implementation are registered globally, and
/// requested at construction.
class Calculator {
public:
    /// Create a new calculator with the given `name` and `parameters`.
    ///
    /// @throws RascalineError if `name` is not associated with a known calculator,
    ///         if `parameters` is not valid JSON, or if `parameters` do not
    ///         contains the expected values for the requested calculator.
    ///
    /// @verbatim embed:rst:leading-slashes
    /// The list of available calculators and the corresponding parameters are
    /// in the :ref:`main documentation <userdoc-references>`. The ``parameters``
    /// should be formatted as JSON, according to the requested calculator
    /// schema.
    /// @endverbatim
    Calculator(std::string name, std::string parameters):
        calculator_(rascal_calculator(name.data(), parameters.data()))
    {
        if (this->calculator_ == nullptr) {
            throw RascalineError(rascal_last_error());
        }
    }

    ~Calculator() {
        details::check_status(rascal_calculator_free(this->calculator_));
    }

    /// Calculator is **NOT** copy-constructible
    Calculator(const Calculator&) = delete;
    /// Calculator can **NOT** be copy-assigned
    Calculator& operator=(const Calculator&) = delete;

    /// Calculator is move-constructible
    Calculator(Calculator&& other) {
        *this = std::move(other);
    }

    /// Calculator can be move-assigned
    Calculator& operator=(Calculator&& other) {
        this->~Calculator();
        this->calculator_ = nullptr;

        std::swap(this->calculator_, other.calculator_);

        return *this;
    }

    /// Get the name used to create this `Calculator`
    std::string name() const {
        auto buffer = std::vector<char>(32, '\0');
        while (true) {
            auto status = rascal_calculator_name(
                calculator_, &buffer[0], buffer.size()
            );

            if (status != RASCAL_BUFFER_SIZE_ERROR) {
                details::check_status(status);
                return std::string(buffer.data());
            }

            // grow the buffer and retry
            buffer.resize(buffer.size() * 2, '\0');
        }
    }

    /// Get the parameters used to create this `Calculator`
    std::string parameters() const {
        auto buffer = std::vector<char>(256, '\0');
        while (true) {
            auto status = rascal_calculator_parameters(
                calculator_, &buffer[0], buffer.size()
            );

            if (status != RASCAL_BUFFER_SIZE_ERROR) {
                details::check_status(status);
                return std::string(buffer.data());
            }

            // grow the buffer and retry
            buffer.resize(buffer.size() * 2, '\0');
        }
    }

    /// Runs a calculation with this calculator on the given ``systems``
    equistore::TensorMap compute(
        std::vector<rascal_system_t>& systems,
        const CalculationOptions& options = CalculationOptions()
    ) const {
        eqs_tensormap_t* descriptor = nullptr;

        details::check_status(rascal_calculator_compute(
            calculator_,
            &descriptor,
            systems.data(),
            systems.size(),
            options.as_rascal_calculation_options_t()
        ));

        return equistore::TensorMap(descriptor);
    }

    /// Runs a calculation for multiple `systems`
    template<typename SystemImpl, typename std::enable_if<std::is_base_of<System, SystemImpl>::value, bool>::type = true>
    equistore::TensorMap compute(
        std::vector<SystemImpl>& systems,
        const CalculationOptions& options = CalculationOptions()
    ) const {
        auto rascal_systems = std::vector<rascal_system_t>();
        for (auto& system: systems) {
            rascal_systems.push_back(system.as_rascal_system_t());
        }

        return this->compute(rascal_systems, std::move(options));
    }

    /// Runs a calculation for a single `system`
    template<typename SystemImpl, typename std::enable_if<std::is_base_of<System, SystemImpl>::value, bool>::type = true>
    equistore::TensorMap compute(
        SystemImpl& system,
        const CalculationOptions& options = CalculationOptions()
    ) const {
        eqs_tensormap_t* descriptor = nullptr;

        auto rascal_system = system.as_rascal_system_t();
        details::check_status(rascal_calculator_compute(
            calculator_,
            &descriptor,
            &rascal_system,
            1,
            options.as_rascal_calculation_options_t()
        ));

        return equistore::TensorMap(descriptor);
    }

    /// Runs a calculation for all the `systems` that where read from a file
    /// using the `BasicSystems(std::string path)` constructor
    equistore::TensorMap compute(
        BasicSystems& systems,
        const CalculationOptions& options = CalculationOptions()
    ) const {
        eqs_tensormap_t* descriptor = nullptr;

        details::check_status(rascal_calculator_compute(
            calculator_,
            &descriptor,
            systems.systems(),
            systems.count(),
            options.as_rascal_calculation_options_t()
        ));

        return equistore::TensorMap(descriptor);
    }

    /// Get the underlying pointer to a `rascal_calculator_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    rascal_calculator_t* as_rascal_calculator_t() {
        return calculator_;
    }

    /// Get the underlying const pointer to a `rascal_calculator_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    const rascal_calculator_t* as_rascal_calculator_t() const {
        return calculator_;
    }

private:
    rascal_calculator_t* calculator_ = nullptr;
};


/// Rascaline uses the [`time_graph`](https://docs.rs/time-graph/) to collect
/// timing information on the calculations. The `Profiler` static class provides
/// access to this functionality.
///
/// The profiling code collects the total time spent inside the most important
/// functions, as well as the function call graph (which function called which
/// other function).
class Profiler {
public:
    /// Enable or disable profiling data collection. By default, data collection
    /// is disabled.
    ///
    /// You can use `Profiler::clear` to reset profiling data to an empty state,
    /// and `Profiler::get` to extract the profiling data.
    ///
    /// @param enabled whether data collection should be enabled or not
    static void enable(bool enabled) {
        details::check_status(rascal_profiling_enable(enabled));
    }

    /// Clear all collected profiling data
    ///
    /// See also `Profiler::enable` and `Profiler::get`.
    static void clear() {
        details::check_status(rascal_profiling_clear());
    }

    /// Extract the current set of data collected for profiling.
    ///
    /// See also `Profiler::enable` and `Profiler::clear`.
    ///
    /// @param format in which format should the data be provided. `"table"`,
    ///              `"short_table"` and `"json"` are currently supported
    /// @returns the current profiling data, in the requested format
    static std::string get(std::string format) {
        auto buffer = std::vector<char>(1024, '\0');
        while (true) {
            auto status = rascal_profiling_get(
                format.c_str(), &buffer[0], buffer.size()
            );

            if (status != RASCAL_BUFFER_SIZE_ERROR) {
                details::check_status(status);
                return std::string(buffer.data());
            }

            // grow the buffer and retry
            buffer.resize(buffer.size() * 2, '\0');
        }
    }

private:
    // make the constructor private and undefined since this class only offers
    // static functions.
    Profiler();
};

}

#endif
