# -*- coding: utf-8 -*-
import json
import ctypes
import numpy as np

from ._rascaline import rascal_system_t, rascal_calculation_options_t, c_uintptr_t
from .clib import _get_library
from .status import _check_rascal_pointer, RascalError
from .descriptor import Descriptor, Indexes
from .systems import wrap_system
from .utils import _call_with_growing_buffer


def _convert_systems(systems):
    try:
        return (rascal_system_t * 1)(wrap_system(systems)._as_rascal_system_t())
    except TypeError:
        # try iterating over the systems
        return (rascal_system_t * len(systems))(
            *list(wrap_system(s)._as_rascal_system_t() for s in systems)
        )


def _options_to_c(use_native_system, samples, features):
    ptr_int32 = ctypes.POINTER(ctypes.c_int32)
    c_options = rascal_calculation_options_t()
    c_options.use_native_system = bool(use_native_system)

    # store data to keep alive here
    c_options.__keepalive = {}

    if samples is None:
        c_options.selected_samples.names = None
        c_options.selected_samples.values = None
        c_options.selected_samples.size = 0
        c_options.selected_samples.count = 0
    else:
        if not isinstance(samples, Indexes):
            raise ValueError(
                "expected selected samples as an `Indexes` instance, "
                f"got {type(samples)} instead"
            )

        samples_names = ctypes.ARRAY(ctypes.c_char_p, len(samples.names))()
        for i, n in enumerate(samples.names):
            samples_names[i] = n.encode("utf8")

        c_options.__keepalive["samples_names"] = samples_names

        c_options.selected_samples.names = samples_names
        c_options.selected_samples.size = len(samples_names)
        c_options.selected_samples.values = samples.ctypes.data_as(ptr_int32)
        c_options.selected_samples.count = samples.shape[0]

    if features is None:
        c_options.selected_features.names = None
        c_options.selected_features.values = None
        c_options.selected_features.size = 0
        c_options.selected_features.count = 0
    else:
        if not isinstance(features, Indexes):
            raise ValueError(
                "expected selected samples as an `Indexes` instance, "
                f"got {type(features)} instead"
            )

        features_names = ctypes.ARRAY(ctypes.c_char_p, len(features.names))()
        for i, n in enumerate(features.names):
            features_names[i] = n.encode("utf8")

        c_options.__keepalive["features_names"] = features_names

        c_options.selected_features.names = features_names
        c_options.selected_features.size = len(features.names)
        c_options.selected_features.values = features.ctypes.data_as(ptr_int32)
        c_options.selected_features.count = features.shape[0]

    return c_options, samples, features


class CalculatorBase:
    def __init__(self, name, parameters):
        self._c_name = name
        self._lib = _get_library()
        parameters = json.dumps(parameters)
        self._as_parameter_ = self._lib.rascal_calculator(
            name.encode("utf8"), parameters.encode("utf8")
        )
        _check_rascal_pointer(self._as_parameter_)

        self._selected_samples = None
        self._selected_features = None

    def __del__(self):
        if hasattr(self, "_lib"):
            # if we failed to load the lib, don't double error by trying to call
            # ``self._lib.rascal_calculator_free``
            self._lib.rascal_calculator_free(self)
        self._as_parameter_ = 0

    @property
    def name(self):
        """The name of this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_name(
                self, buffer, bufflen
            )
        )

    @property
    def c_name(self):
        """The name used to register & create this calculator"""
        return self._c_name

    @property
    def parameters(self):
        """The parameters (formatted as JSON) used to create this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_parameters(
                self, buffer, bufflen
            )
        )

    def features_count(self):
        """
        Get the default number of features this calculator will produce.

        This number corresponds to the size of second dimension of the
        ``values`` and ``gradients`` arrays of the
        :py:class:`rascaline.Descriptor` returned by
        :py:func:`rascaline.calculators.CalculatorBase.compute`.
        """
        size = c_uintptr_t()
        self._lib.rascal_calculator_features_count(self, size)
        return size.value

    def compute(
        self,
        systems,
        descriptor=None,
        use_native_system=True,
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

        :param bool use_native_system: defaults to ``True``. If ``True``, copy
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

        :type selected_samples: Optional[:py:class:`rascaline.descriptor.Indexes`]

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

        :type selected_features: Optional[:py:class:`rascaline.descriptor.Indexes`]

        :return: the ``descriptor`` parameter or the new new descriptor if
                 ``descriptor`` was ``None``.
        """

        if descriptor is None:
            descriptor = Descriptor()

        c_systems = _convert_systems(systems)

        # keep the selected samples & features arrays in scope to prevent them
        # from being garbage collected
        c_options, samples, features = _options_to_c(
            use_native_system=use_native_system,
            samples=selected_samples,
            features=selected_features,
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
    """

    def __init__(self, cutoff, max_neighbors):
        parameters = {"cutoff": cutoff, "max_neighbors": max_neighbors}
        super().__init__("sorted_distances", parameters)


class SphericalExpansion(CalculatorBase):
    """
    The spherical expansion is at the core of representations in the SOAP
    (Smooth Overlap of Atomic Positions) family of descriptors. The spherical
    expansion represent atomic density as a collection of gaussian functions
    centered on each atom, and then represent the local density around each atom
    (up to a cutoff) on a basis of radial functions and spherical harmonics.
    This representation is not rotationally invariant, for that you should use
    the :py:class:`SoapPowerSpectrum` class.

    See `this review article <https://doi.org/10.1063/1.5090481>`_ for more
    information on the SOAP representation, and `this paper
    <https://doi.org/10.1063/5.0044689>`_ for information on how it is
    implemented in rascaline.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <spherical-expansion>`.
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
    """
    The SOAP power spectrum is the main member of the SOAP (Smooth Overlap of
    Atomic Positions) family of descriptors. The power spectrum is based on the
    :py:class:`SphericalExpansion` coefficients, which are combined to create a
    rotationally invariant three-body descriptor.

    See `this review article <https://doi.org/10.1063/1.5090481>`_ for more
    information on the SOAP representation, and `this paper
    <https://doi.org/10.1063/5.0044689>`_ for information on how it is
    implemented in rascaline.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <soap-power-spectrum>`.
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
        super().__init__("soap_power_spectrum", parameters)
