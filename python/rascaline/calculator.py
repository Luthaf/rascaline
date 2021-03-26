# -*- coding: utf-8 -*-
import json
import ctypes
import numpy as np

from ._rascaline import rascal_system_t, rascal_status_t, rascal_calculation_options_t
from .clib import _get_library
from .status import _check_rascal_pointer, RascalError
from .descriptor import Descriptor
from .systems import wrap_system


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except RascalError as e:
            if (
                e.status == rascal_status_t.RASCAL_INVALID_PARAMETER_ERROR.value
                and "string buffer is not big enough" in e.args[0]
            ):
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def _check_selected_indexes(indexes, kind):
    if len(indexes.shape) != 2:
        raise ValueError(f"selected {kind} array must be a two-dimensional array")

    if not np.can_cast(indexes.dtype, np.float64, "safe"):
        raise ValueError(f"selected {kind} array must contain float64 values")


def _convert_systems(systems):
    try:
        return (rascal_system_t * 1)(wrap_system(systems)._as_rascal_system_t())
    except TypeError:
        # try iterating over the systems
        return (rascal_system_t * len(systems))(
            *list(wrap_system(s)._as_rascal_system_t() for s in systems)
        )


def _options_to_c(**options):
    known_options = [
        "use_native_system",
        "selected_samples",
        "selected_features",
    ]
    for option in options.keys():
        if option not in known_options:
            raise Exception(f"unknown option '{option}' passed to compute")

    samples = options.get("selected_samples")
    if samples is not None:
        samples = np.array(samples)
        if samples.dtype.fields is not None:
            # convert structured array back to int32 array
            size = len(samples)
            samples = samples.view(dtype=np.int32).reshape((size, -1))
        else:
            _check_selected_indexes(samples, "samples")
            samples = np.array(samples, dtype=np.int32)

    features = options.get("selected_features")
    if features is not None:
        features = np.array(features)
        if features.dtype.fields is not None:
            # convert structured array back to int32 array
            size = len(features)
            features = features.view(dtype=np.int32).reshape((size, -1))
        else:
            _check_selected_indexes(features, "features")
            features = np.array(features, dtype=np.int32)

    ptr_int32 = ctypes.POINTER(ctypes.c_int32)
    c_options = rascal_calculation_options_t()
    c_options.use_native_system = bool(options.get("use_native_system", False))

    if samples is None:
        c_options.selected_samples = None
        c_options.selected_samples_count = 0
    else:
        c_options.selected_samples = samples.ctypes.data_as(ptr_int32)
        c_options.selected_samples_count = samples.size

    if features is None:
        c_options.selected_features = None
        c_options.selected_features_count = 0
    else:
        c_options.selected_features = features.ctypes.data_as(ptr_int32)
        c_options.selected_features_count = features.size

    return c_options


class CalculatorBase:
    def __init__(self, name, parameters):
        self._lib = _get_library()
        parameters = json.dumps(parameters)
        self._as_parameter_ = self._lib.rascal_calculator(
            name.encode("utf8"), parameters.encode("utf8")
        )
        _check_rascal_pointer(self._as_parameter_)

    def __del__(self):
        self._lib.rascal_calculator_free(self)
        self._as_parameter_ = 0

    @property
    def name(self):
        """The name used to register this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_name(
                self, buffer, bufflen
            )
        )

    @property
    def parameters(self):
        """The parameters (formatted as JSON) used to create this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_parameters(
                self, buffer, bufflen
            )
        )

    def compute(
        self,
        systems,
        descriptor=None,
        use_native_system=False,
        selected_samples=None,
        selected_features=None,
    ):
        """
        Run a calculation with this calculator on the given ``systems``, storing
        the resulting data in the ``descriptor``.

        :param systems: single system or list of systems on which to run the
                        calculation. Multiple types of systems are supported,
                        see the documentation for
                        :py:func:`rascaline.systems.wrap_system` to get the full
                        list.

        :param descriptor: Descriptor in which the result of the calculation are
                           stored. If this parameter is ``None``, a new
                           desriptor is created and returned by this function.

        :type descriptor: :py:class:`Descriptor`, optional

        :param bool use_native_system: defaults to ``False``. If ``True``, copy
            data from the ``systems`` into Rust ``SimpleSystem``. This can be a
            lot faster than having to cross the FFI boundary often when acessing
            the neighbor list.

        :param selected_samples: defaults to ``None``. List of samples on which
            to run the calculation. Use ``None`` to run the calculation on all
            samples in the ``systems`` (this is the default).

            This should be either a numpy ndarray with ``dtype=np.int32`` and
            two dimensions; or a slice of a :py:class:`rascaline.descriptor.Indexes`
            instance extracted from a calculator. If a raw ndarray is used, the
            first dimension of the array is the list of all samples to consider;
            and the second dimension of the array must match the size of the
            sample indexes used by this calculator. Each row of the array
            describes a single sample and will be validated by the calculator.

        :type selected_samples: Optional[numpy.ndarray | :py:class:`rascaline.descriptor.Indexes`]

        :param selected_features: defaults to ``None``. List of features on
            which to run the calculation. Use ``None`` to run the calculation on
            all features (this is the default).

            This should be either a numpy ndarray with ``dtype=np.int32`` and
            two dimensions; or a slice of a :py:class:`rascaline.descriptor.Indexes`
            instance extracted from a calculator. If a raw ndarray is used, the
            first dimension of the array is the list of all features to
            consider; and the second dimension of the array must match the size
            of the features used by this calculator.  Each row of the array
            describes a single feature and will be validated by the calculator.

        :type selected_features: Optional[numpy.ndarray | :py:class:`rascaline.descriptor.Indexes`]

        :return: the ``descriptor`` parameter or the new new descriptor if
                 ``descriptor`` was ``None``.
        """

        if descriptor is None:
            descriptor = Descriptor()

        c_systems = _convert_systems(systems)
        c_options = _options_to_c(
            use_native_system=use_native_system,
            selected_samples=selected_samples,
            selected_features=selected_features,
        )
        self._lib.rascal_calculator_compute(
            self, descriptor, c_systems, c_systems._length_, c_options
        )
        return descriptor


class DummyCalculator(CalculatorBase):
    def __init__(self, cutoff, delta, name, gradients):
        parameters = {
            "cutoff": cutoff,
            "delta": delta,
            "name": name,
            "gradients": gradients,
        }
        super().__init__("dummy_calculator", parameters)


class SortedDistances(CalculatorBase):
    """
    Sorted distances vector representation of an atomic environment.

    Each atomic center is represented by a vector of distance to its neighbors
    within the spherical ``cutoff``, sorted from smallest to largest. If there
    are less neighbors than ``max_neighbors``, the remaining entries are filled
    with ``cutoff`` instead.

    Separate species for neighbors are represented separately, meaning that the
    ``max_neighbors`` parameter only apply to a single species.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <sorted-distances>`.

    This class inherits from :py:class:`rascaline.calculator.CalculatorBase` and
    exposes the same functions and properties.
    """

    def __init__(self, cutoff, max_neighbors):
        parameters = {"cutoff": cutoff, "max_neighbors": max_neighbors}
        super().__init__("sorted_distances", parameters)


class SphericalExpansion(CalculatorBase):
    """
    The spherical expansion is at the core of representations in the SOAP
    (Smooth Overlap of Atomic Positions) family. See `this review article
    <https://doi.org/10.1063/1.5090481>`_ for more information on the SOAP
    representation, and `this paper <https://doi.org/10.1063/5.0044689>`_ for
    information on how it is implemented in rascaline.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <spherical-expansion>`.

    This class inherits from :py:class:`rascaline.calculator.CalculatorBase` and
    exposes the same functions and properties.
    """

    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        radial_basis,
        gradients,
        cutoff_function,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "radial_basis": radial_basis,
            "gradients": gradients,
            "cutoff_function": cutoff_function,
        }
        super().__init__("spherical_expansion", parameters)


class SoapPowerSpectrum(CalculatorBase):
    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        radial_basis,
        gradients,
        cutoff_function,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "radial_basis": radial_basis,
            "gradients": gradients,
            "cutoff_function": cutoff_function,
        }
        super().__init__("soap_power_spectrum", parameters)
