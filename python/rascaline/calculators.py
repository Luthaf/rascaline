# -*- coding: utf-8 -*-
import ctypes
import json
from typing import List, Optional, Union

from equistore import Labels, TensorMap
from equistore._c_api import eqs_tensormap_t

from ._c_api import rascal_calculation_options_t, rascal_system_t
from ._c_lib import _get_library
from .status import _check_rascal_pointer
from .systems import IntoSystem, wrap_system
from .utils import _call_with_growing_buffer


def _convert_systems(systems):
    try:
        return (rascal_system_t * 1)(wrap_system(systems)._as_rascal_system_t())
    except TypeError:
        # try iterating over the systems
        return (rascal_system_t * len(systems))(
            *list(wrap_system(s)._as_rascal_system_t() for s in systems)
        )


def _options_to_c(
    use_native_system,
    selected_samples,
    selected_properties,
):
    c_options = rascal_calculation_options_t()
    c_options.use_native_system = bool(use_native_system)

    # store data to keep alive here
    c_options.__keepalive = {}

    if selected_samples is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_samples, Labels):
        selected_samples = selected_samples._as_eqs_labels_t()
        c_options.selected_samples.subset = ctypes.pointer(selected_samples)
        c_options.__keepalive["selected_samples"] = selected_samples
    elif isinstance(selected_samples, TensorMap):
        c_options.selected_samples.predefined = selected_samples._ptr
    else:
        raise ValueError(
            "expected selected samples to be either an `equistore.Labels` "
            "instance, or an got `equistore.TensorMap` instance, got "
            f"{type(selected_samples)} instead"
        )

    if selected_properties is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_properties, Labels):
        selected_properties = selected_properties._as_eqs_labels_t()
        c_options.selected_properties.subset = ctypes.pointer(selected_properties)
        c_options.__keepalive["selected_properties"] = selected_properties
    elif isinstance(selected_properties, TensorMap):
        c_options.selected_properties.predefined = selected_properties._ptr
    else:
        raise ValueError(
            "expected selected properties to be either an `equistore.Labels` "
            "instance, or an got `equistore.TensorMap` instance, got "
            f"{type(selected_properties)} instead"
        )

    return c_options


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
        """Name of this calculator."""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_name(
                self, buffer, bufflen
            )
        )

    @property
    def c_name(self):
        """Name used to register & create this calculator."""
        return self._c_name

    @property
    def parameters(self):
        """Parameters (formatted as JSON) used to create this calculator."""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.rascal_calculator_parameters(
                self, buffer, bufflen
            )
        )

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        use_native_system: bool = True,
        selected_samples: Optional[Union[Labels, TensorMap]] = None,
        selected_properties: Optional[Union[Labels, TensorMap]] = None,
    ) -> TensorMap:
        """Runs a calculation with this calculator on the given ``systems``.

        :param systems: single system or list of systems on which to run the
            calculation. The systems will automatically be wrapped into
            compatible classes (using :py:func:`rascaline.systems.wrap_system`).
            Multiple types of systems are supported, see the documentation of
            :py:class:`rascaline.IntoSystem` to get the full list.

        :param use_native_system: If ``True`` (this is the default), copy data
            from the ``systems`` into Rust ``SimpleSystem``. This can be a lot
            faster than having to cross the FFI boundary often when accessing
            the neighbor list. Otherwise the Python neighbor list is used.

        :param selected_samples: Set of samples on which to run the calculation.
            Use ``None`` to run the calculation on all samples in the
            ``systems`` (this is the default).

            If the :py:class:`equistore.Labels` contains the same variables as
            the default set of samples for this calculator, then only entries
            from the full set that also appear in this selection will be used.

            If the labels contains a subset of the variables of the full set of
            samples, then only entries from the full set which match one of the
            entry in this selection for all of the selection variable will be
            used. TODO

        :param selected_properties: Set of properties to compute. Use ``None``
            to run the calculation on all properties (this is the default).

            If the :py:class:`equistore.Labels` contains the same variables as
            the default set of properties for this calculator, then only entries
            from the full set that also appear in this selection will be used.

            If the labels contains a subset of the variables of the full set of
            properties, then only entries from the full set which match one of
            the entry in this selection for all of the selection variable will
            be used. TODO
        """

        c_systems = _convert_systems(systems)
        tensor_map_ptr = ctypes.POINTER(eqs_tensormap_t)()

        c_options = _options_to_c(
            use_native_system=use_native_system,
            selected_samples=selected_samples,
            selected_properties=selected_properties,
        )
        self._lib.rascal_calculator_compute(
            self, tensor_map_ptr, c_systems, c_systems._length_, c_options
        )

        return TensorMap._from_ptr(tensor_map_ptr)


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
    """Sorted distances vector representation of an atomic environment.

    Each atomic center is represented by a vector of distance to its neighbors
    within the spherical ``cutoff``, sorted from smallest to largest. If there
    are less neighbors than ``max_neighbors``, the remaining entries are filled
    with ``cutoff`` instead.

    Separate species for neighbors are represented separately, meaning that the
    ``max_neighbors`` parameter only apply to a single species.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <sorted-distances>`.
    """

    def __init__(self, cutoff, max_neighbors, separate_neighbor_species):
        parameters = {
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
            "separate_neighbor_species": separate_neighbor_species,
        }
        super().__init__("sorted_distances", parameters)


class SphericalExpansion(CalculatorBase):
    """Spherical expansion of Smooth Overlap of Atomic Positions (SOAP).

    The spherical expansion is at the core of representations in the SOAP
    family of descriptors. The spherical
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
        center_atom_weight,
        gradients,
        cutoff_function,
        radial_scaling=None,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "center_atom_weight": center_atom_weight,
            "radial_basis": radial_basis,
            "gradients": gradients,
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("spherical_expansion", parameters)


class SoapPowerSpectrum(CalculatorBase):
    """Power spectrumm of Smooth Overlap of Atomic Positions (SOAP).

    The SOAP power spectrum is the main member of the SOAP
    family of descriptors. The power spectrum is based on the
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
        center_atom_weight,
        radial_basis,
        gradients,
        cutoff_function,
        radial_scaling=None,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "center_atom_weight": center_atom_weight,
            "radial_basis": radial_basis,
            "gradients": gradients,
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("soap_power_spectrum", parameters)
