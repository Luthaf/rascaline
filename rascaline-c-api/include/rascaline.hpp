#ifndef RASCALINE_HPP
#define RASCALINE_HPP

#include <array>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "rascaline.h"

/// This file contains the C++ API to rascaline, manually built on top of the C
/// API defined in `rascaline.h`. This API uses the standard C++ library where
/// convenient, but also allow to drop back to the C API if required, by
/// providing functions to extract the C API handles (named `as_rascal_XXX`).


namespace rascaline {

/// Exception class for all error thrown by rascaline
class RascalError : public std::runtime_error {
public:
    /// Create a new error with the given message
    RascalError(std::string message): std::runtime_error(message) {}
    ~RascalError() = default;

    /// RascalError is copy-constructible
    RascalError(const RascalError&) = default;
    /// RascalError is move-constructible
    RascalError(RascalError&&) = default;
    /// RascalError can be copy-assigned
    RascalError& operator=(const RascalError&) = default;
    /// RascalError can be move-assigned
    RascalError& operator=(RascalError&&) = default;
};

namespace details {
    /// Check the status returned by a rascal function, throwing an exception
    /// with the latest error message if the status is not `RASCAL_SUCCESS`.
    inline void check_status(rascal_status_t status) {
        if (status != RASCAL_SUCCESS) {
            throw RascalError(rascal_last_error());
        }
    }
}

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
    virtual const uintptr_t* species() const = 0;

    /// Get a pointer to the first element of a contiguous array containing the
    /// atomic cartesian coordinates. `positions[0], positions[1], positions[2]`
    /// must contain the x, y, z cartesian coordinates of the first atom, and so
    /// on. The array should contain `3 x System::size()` elements.
    virtual const double* positions() const = 0;

    /// Unit cell representation as a 3x3 matrix
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
                *size = static_cast<const System*>(self)->size();
            },
            // species
            [](const void* self, const uintptr_t** species) {
                *species = static_cast<const System*>(self)->species();
            },
            // positions
            [](const void* self, const double** positions) {
                *positions = (reinterpret_cast<const System*>(self))->positions();
            },
            // cell
            [](const void* self, double* cell) {
                auto cpp_cell = reinterpret_cast<const System*>(self)->cell();
                std::memcpy(cell, &cpp_cell[0][0], 9 * sizeof(double));
            },
            // compute_neighbors
            [](void* self, double cutoff) {
                reinterpret_cast<System*>(self)->compute_neighbors(cutoff);
            },
            // pairs
            [](const void* self, const rascal_pair_t** pairs, uintptr_t* size) {
                const auto& cpp_pairs = reinterpret_cast<const System*>(self)->pairs();
                *pairs = cpp_pairs.data();
                *size = cpp_pairs.size();
            },
            // pairs_containing
            [](const void* self, uintptr_t center, const rascal_pair_t** pairs, uintptr_t* size) {
                const auto& cpp_pairs = reinterpret_cast<const System*>(self)->pairs_containing(center);
                *pairs = cpp_pairs.data();
                *size = cpp_pairs.size();
            }
        };
    }
};


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
    /// @throws RascalError if chemfiles can not read the file
    BasicSystems(std::string path): systems_(nullptr), count_(0) {
        details::check_status(rascal_basic_systems_read(path.c_str(), &systems_, &count_));
    }

    ~BasicSystems() {
        details::check_status(rascal_basic_systems_free(systems_, count_));
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
    rascal_system_t* systems_;
    uintptr_t count_;
};


/// An `ArrayView` is view inside a rust-owned 2D array, similar to std::span
/// for 2D arrays.
///
/// Instances of this class do not own their memory, but rather point inside
/// memory owned by the Rust library. For this reason, they are only valid to
/// use while the corresponding memory is not touched. In practice, this means
/// that calling `Calculator::compute` with a descriptor will invalidate all
/// `ArrayView` created from this `Descriptor`.
template<typename T>
class ArrayView {
public:
    /// Create a new empty `ArrayView`, with shape `[0, 0]`.
    ArrayView(): ArrayView(nullptr, {0, 0}) {}

    /// Create a new `ArrayView` with the given `shape`. `data` must point to
    /// contiguous memory containing `shape[0] x shape[1]` elements, to be
    /// interpreted as a 2D array in row-major order. The resulting `ArrayView`
    /// is only valid for as long as `data` is.
    ///
    /// This function is intended for internal use only.
    ArrayView(const T* data, std::array<size_t, 2> shape): data_(data), shape_(shape) {
        if (shape_[0] != 0 && shape_[1] != 0) {
            if (data_ == nullptr) {
                throw RascalError("invalid parameters to ArrayView, got null data pointer and non zero size");
            }
        }

        static_assert(
            std::is_arithmetic<T>::value,
            "ArrayView only works with integers and floating points"
        );
    }

    ~ArrayView() {
        // no memory to release
    }

    /// ArrayView is copy-constructible
    ArrayView(const ArrayView&) = default;
    /// ArrayView is move-constructible
    ArrayView(ArrayView&&) = default;
    /// ArrayView can be copy-assigned
    ArrayView& operator=(const ArrayView&) = default;
    /// ArrayView can be move-assigned
    ArrayView& operator=(ArrayView&&) = default;

    /// Get a the value inside this `ArrayView` at index `i, j`
    T operator()(size_t i, size_t j) const {
        assert(i < shape_[0] && j < shape_[1]);
        return data_[i * shape_[1] + j];
    }

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    const T* data() const {
        return data_;
    }

    /// Get the shape of this array
    std::array<size_t, 2> shape() const {
        return shape_;
    }

    /// Check if this array is empty, i.e. if at least one of the shape element
    /// is 0.
    bool is_empty() const {
        return shape_[0] == 0 && shape_[1] == 0;
    }

private:
    const T* data_;
    std::array<size_t, 2> shape_;
};


/// List of selected indexes to specify in a `CalculationOptions`
///
/// Indexes represents either samples or features. By default, the calculation
/// run for all samples and all features. By constructing a `SelectedIndexes`
/// with the right values, users can specifically ask to only run the
/// calculation on a subset of samples and/or features.
///
/// The easiest way to create a `SelectedIndexes` is through `Indexes::selected`.
class SelectedIndexes {
public:
    /// Create a `SelectedIndexes` indicating that we want to run the
    /// calculation on all possible indexes, leaving the determination of these
    /// indexes to the calculator class.
    SelectedIndexes(): SelectedIndexes(0) {}

    /// Create an empty set of `SelectedIndexes` with the given `size`. The
    /// `size` must match the second dimension of the expected indexes (sample
    /// or features).
    SelectedIndexes(size_t size): size_(size), indexes_() {}

    ~SelectedIndexes() = default;
    /// SelectedIndexes is copy-constructible
    SelectedIndexes(const SelectedIndexes&) = default;
    /// SelectedIndexes is move-constructible
    SelectedIndexes(SelectedIndexes&&) = default;
    /// SelectedIndexes can be copy-assigned
    SelectedIndexes& operator=(const SelectedIndexes&) = default;
    /// SelectedIndexes can be move-assigned
    SelectedIndexes& operator=(SelectedIndexes&&) = default;

    /// Add a single set of indexes to the selected indexes. The number of
    /// elements in `indexes` must match the size passed when constructing this
    /// `SelectedIndexes`.
    ///
    /// @throws RascalError if the size of indexes do not match the expected one
    void add(const std::vector<int32_t>& indexes) {
        this->add(indexes.begin(), indexes.end());
    }

    /// Add a single set of indexes (between iterator `begin` and `end`) to the
    /// selected indexes. The number of elements between `begin` and `end` must
    /// match the size passed when constructing this `SelectedIndexes`.
    ///
    /// @throws RascalError if the size of indexes do not match the expected one
    template<typename Iterator>
    void add(Iterator begin, Iterator end) {
        auto iterator_size = std::distance(begin, end);
        if (iterator_size != size_) {
            throw RascalError(
                "invalid size for new selected vector, expected " +
                std::to_string(size_) + " got " + std::to_string(iterator_size)
            );
        }
        indexes_.reserve(indexes_.size() + size_);
        indexes_.insert(indexes_.end(), begin, end);
    }

    /// Get a pointer to the first element of the underlying array.
    ///
    /// This function is intended for internal use only.
    const int32_t* data() const {
        return indexes_.data();
    }

    /// Get the number of indexes in this array. Each index is a vector of size
    /// `Indexes.size`.
    size_t count() const {
        return indexes_.size() / size_;
    }

    /// Get the size of the indexes in this array.
    size_t size() const {
        return size_;
    }

private:
    size_t size_;
    std::vector<int32_t> indexes_;
};


/// A set of `Indexes` contains metdata describing row or columns in the
/// `values` and `gradients` arrays in a `Descriptor`.
///
/// Each instance of `Indexes` contains both a 2D array of shape `count x size`
/// and a vector of names associated with the columns of the 2D array.
class Indexes: public ArrayView<int32_t> {
public:
    /// Create a new Indexes with the given `names` and corresponding `data`
    ///
    /// This function is intended for internal use only.
    Indexes(std::vector<std::string> names, ArrayView<int32_t> data):
        ArrayView<int32_t>(std::move(data)), names_(names)
    {
        assert(this->shape()[1] == names_.size());
    }

    /// Get the names of variables described by this `Indexes` set.
    const std::vector<std::string>& names() const {
        return names_;
    }

    /// Construct a `SelectedIndexes` containing only values from the requested
    /// `indexes`.
    SelectedIndexes select(std::vector<size_t> indexes) {
        auto size = this->shape()[1];
        auto selected = SelectedIndexes(size);
        auto data = this->data();
        for (auto i: indexes) {
            selected.add(data + i * size, data + (i + 1) * size);
        }
        return selected;
    }

private:
    std::vector<std::string> names_;
};

/// Descriptors store the result of a single calculation on a set of systems.
///
/// They contains the values produced by the calculation; as well as metdata to
/// interpret these values. In particular, it contains two additional arrays
/// describing the `samples` (associated with rows in the values) and `features`
/// (associated with columns in the values).
///
/// Optionally, a descriptor can also contain gradients of the samples. In this
/// case, some additional metadata is available to describe the rows of the
/// gradients array in `gradients_samples`. The columns of the gradients array
/// are still described by the same `features`.
class Descriptor final {
public:
    /// Create a new empty descriptor.
    Descriptor(): descriptor_(rascal_descriptor()) {
        if (this->descriptor_ == nullptr) {
            throw RascalError(rascal_last_error());
        }
    }

    ~Descriptor() {
        details::check_status(rascal_descriptor_free(this->descriptor_));
    }

    /// Descriptor is **NOT** copy-constructible
    Descriptor(const Descriptor&) = delete;
    /// Descriptor can **NOT** be copy-assigned
    Descriptor& operator=(const Descriptor&) = delete;

    /// Descriptor is move-constructible
    Descriptor(Descriptor&& other) {
        *this = std::move(other);
    }

    /// Descriptor can be move-assigned
    Descriptor& operator=(Descriptor&& other) {
        this->descriptor_ = other.descriptor_;
        other.descriptor_ = nullptr;
        return *this;
    }

    /// Get the values stored inside this descriptor after a call to
    /// `Calculator::compute`.
    ArrayView<double> values() const {
        const double* data = nullptr;
        uintptr_t samples = 0;
        uintptr_t features = 0;
        details::check_status(rascal_descriptor_values(
            descriptor_, &data, &samples, &features
        ));

        return ArrayView<double>(data, {samples, features});
    }

    /// Get the values stored inside this descriptor after a call to
    /// `Calculator::compute`, if any.
    ///
    /// If this descriptor does not contain gradient data, an empty array is
    /// returned.
    ArrayView<double> gradients() const {
        const double* data = nullptr;
        uintptr_t samples = 0;
        uintptr_t features = 0;
        details::check_status(rascal_descriptor_gradients(
            descriptor_, &data, &samples, &features
        ));

        return ArrayView<double>(data, {samples, features});
    }

    /// Get metdata describing the samples/rows in `Descriptor::values`.
    ///
    /// This is stored as a **read only** 2D array, where each column is named.
    Indexes samples() const {
        return this->indexes(RASCAL_INDEXES_SAMPLES);
    }

    /// Get metdata describing the features/columns in `Descriptor::values` and
    /// `Descriptor::gradients`.
    ///
    /// This is stored as a **read only** 2D array, where each column is named.
    Indexes features() const {
        return this->indexes(RASCAL_INDEXES_FEATURES);
    }

    /// Get metdata describing the gradients rows in `Descriptor::gradients`.
    ///
    /// This is stored as a **read only** 2D array, where each column is named.
    Indexes gradients_samples() const {
        return this->indexes(RASCAL_INDEXES_GRADIENT_SAMPLES);
    }

    /// Move the values with names in ``variables`` from samples to features.
    void densify(std::vector<std::string> variables) {
        auto c_variables = std::vector<const char*>(variables.size());
        for (size_t i=0; i<variables.size(); i++) {
            c_variables[i] = variables[i].data();
        }
        details::check_status(
            rascal_descriptor_densify(descriptor_, c_variables.data(), variables.size())
        );
    }

    /// Get the underlying pointer to a `rascal_descriptor_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    rascal_descriptor_t* as_rascal_descriptor_t() {
        return descriptor_;
    }

    /// Get the underlying const pointer to a `rascal_descriptor_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    const rascal_descriptor_t* as_rascal_descriptor_t() const {
        return descriptor_;
    }

private:
    /// Generic function to get a set of `Indexes` out of this descriptor.
    Indexes indexes(rascal_indexes indexes) const {
        const int32_t* data = nullptr;
        uintptr_t count = 0;
        uintptr_t size = 0;
        details::check_status(rascal_descriptor_indexes(
            descriptor_, indexes, &data, &count, &size
        ));

        auto array = ArrayView<int32_t>(data, {count, size});

        auto names = std::vector<std::string>();
        if (size != 0) {
            auto c_names = std::vector<const char*>(size, nullptr);
            details::check_status(rascal_descriptor_indexes_names(
                descriptor_, indexes, c_names.data(), size
            ));
            for (const auto name: c_names) {
                names.push_back(std::string(name));
            }
        }

        return Indexes(std::move(names), std::move(array));
    }

    rascal_descriptor_t* descriptor_;
};


/// Options that can be set to change how a calculator operates.
class CalculationOptions {
public:
    /// Copy the data from systems into native `SimpleSystem`. This can be
    /// faster than having to cross the FFI boundary too often.
    bool use_native_system = false;

    /// List of samples on which to run the calculation. Use an empty array  to
    /// run the calculation on all samples. If necessary, gradients samples
    /// will be derived from the values given in `selected_samples`.
    SelectedIndexes selected_samples = SelectedIndexes();

    /// List of features on which to run the calculation. Use an empty array  to
    /// run the calculation on all features.
    SelectedIndexes selected_features = SelectedIndexes();

    /// Convert this instance of `CalculationOptions` to a
    /// `rascal_calculation_options_t`.
    ///
    /// This is an advanced function that most users don't need to call
    /// directly.
    rascal_calculation_options_t as_rascal_calculation_options_t() const {
        auto options = rascal_calculation_options_t{};
        options.use_native_system = this->use_native_system;

        if (this->selected_samples.size() == 0) {
            options.selected_samples = nullptr;
            options.selected_samples_count = 0;
        } else {
            options.selected_samples = this->selected_samples.data();
            options.selected_samples_count = this->selected_samples.count();
            options.selected_samples_count *= this->selected_samples.size();
        }

        if (this->selected_features.size() == 0) {
            options.selected_features = nullptr;
            options.selected_features_count = 0;
        } else {
            options.selected_features = this->selected_features.data();
            options.selected_features_count = this->selected_features.count();
            options.selected_features_count *= this->selected_features.size();
        }

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
    /// @throws RascalError if `name` is not associated with a known calculator,
    ///         if `parameters` is not valid JSON, or if `parameters` do not
    ///         contains the expected values for the requested calculator.
    ///
    /// @verbatim embed:rst:leading-slashes
    /// The list of available calculators and the corresponding parameters are
    /// in the :ref:`main documentation <calculators-list>`. The ``parameters``
    /// should be formatted as JSON, according to the requested calculator
    /// schema.
    /// @endverbatim
    Calculator(std::string name, std::string parameters):
        calculator_(rascal_calculator(name.data(), parameters.data()))
    {
        if (this->calculator_ == nullptr) {
            throw RascalError(rascal_last_error());
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
        this->calculator_ = other.calculator_;
        other.calculator_ = nullptr;
        return *this;
    }

    /// Get the name used to create this `Calculator`
    std::string name() const {
        auto buffer = std::vector<char>(32, '\0');
        while (true) {
            auto status = rascal_calculator_name(
                calculator_, &buffer[0], buffer.size()
            );

            if (status != RASCAL_INVALID_PARAMETER_ERROR) {
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

            if (status != RASCAL_INVALID_PARAMETER_ERROR) {
                details::check_status(status);
                return std::string(buffer.data());
            }

            // grow the buffer and retry
            buffer.resize(buffer.size() * 2, '\0');
        }
    }

    /// Run a calculation with this `calculator` on the given `systems`, storing
    /// the resulting data in the `descriptor`. Options for this calculation can
    /// be passed in `options`.
    void compute(std::vector<System*> systems, Descriptor& descriptor, CalculationOptions options = CalculationOptions()) const {
        auto rascal_systems = std::vector<rascal_system_t>();
        for (auto& system: systems) {
            assert(system != nullptr);
            rascal_systems.push_back(system->as_rascal_system_t());
        }

        details::check_status(rascal_calculator_compute(
            calculator_,
            descriptor.as_rascal_descriptor_t(),
            rascal_systems.data(),
            rascal_systems.size(),
            options.as_rascal_calculation_options_t()
        ));
    }

    /// Run a calculation with this `calculator` on the given `systems`, and
    /// return the resulting data in a new `Descriptor`. Options for this
    /// calculation can be passed in `options`.
    Descriptor compute(std::vector<System*> systems, CalculationOptions options = CalculationOptions()) const {
        auto descriptor = Descriptor();
        this->compute(std::move(systems), descriptor, options);
        return descriptor;
    }

    /// Run a calculation with this `calculator` on the given `systems`, storing
    /// the resulting data in the `descriptor`. Options for this calculation can
    /// be passed in `options`.
    void compute(BasicSystems systems, Descriptor& descriptor, CalculationOptions options = CalculationOptions()) const {
        details::check_status(rascal_calculator_compute(
            calculator_,
            descriptor.as_rascal_descriptor_t(),
            systems.systems(),
            systems.count(),
            options.as_rascal_calculation_options_t()
        ));
    }

    /// Run a calculation with this `calculator` on the given `systems`, and
    /// return the resulting data in a new `Descriptor`. Options for this
    /// calculation can be passed in `options`.
    Descriptor compute(BasicSystems systems, CalculationOptions options = CalculationOptions()) const {
        auto descriptor = Descriptor();
        this->compute(std::move(systems), descriptor, options);
        return descriptor;
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
    rascal_calculator_t* calculator_;
};

}

#endif
