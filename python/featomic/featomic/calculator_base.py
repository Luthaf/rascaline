import ctypes
from typing import List, Optional, Union

from metatensor import Labels, TensorMap
from metatensor._c_api import c_uintptr_t, mts_tensormap_t

from ._c_api import (
    FEATOMIC_BUFFER_SIZE_ERROR,
    featomic_calculation_options_t,
    featomic_system_t,
)
from ._c_lib import _get_library
from .status import FeatomicError, _check_featomic_pointer
from .systems import IntoSystem, wrap_system


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except FeatomicError as e:
            if e.status == FEATOMIC_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def _convert_systems(systems):
    try:
        return (featomic_system_t * 1)(wrap_system(systems)._as_featomic_system_t())
    except TypeError:
        # try iterating over the systems
        return (featomic_system_t * len(systems))(
            *list(wrap_system(s)._as_featomic_system_t() for s in systems)
        )


def _options_to_c(
    gradients,
    use_native_system,
    selected_samples,
    selected_properties,
    selected_keys,
):
    if gradients is None:
        gradients = []

    if not isinstance(gradients, list):
        raise ValueError(
            f"`gradients` parameter must be a list of str, got a {type(gradients)}"
        )
    for parameter in gradients:
        if not isinstance(parameter, str):
            raise ValueError(
                "`gradients` parameter must be a list of str, got a "
                f"{type(parameter)} in the list"
            )

    c_gradients = ctypes.ARRAY(ctypes.c_char_p, len(gradients))()
    for i, parameter in enumerate(gradients):
        c_gradients[i] = parameter.encode("utf8")

    c_options = featomic_calculation_options_t()
    c_options.gradients = c_gradients
    c_options.gradients_count = c_gradients._length_
    c_options.use_native_system = bool(use_native_system)

    # store data to keep alive here
    c_options.__keepalive = {}

    if selected_samples is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_samples, Labels):
        selected_samples = selected_samples._as_mts_labels_t()
        c_options.selected_samples.subset = ctypes.pointer(selected_samples)
        c_options.__keepalive["selected_samples"] = selected_samples
    elif isinstance(selected_samples, TensorMap):
        c_options.selected_samples.predefined = selected_samples._ptr
    else:
        raise ValueError(
            "expected selected samples to be either an `metatensor.Labels` "
            "instance, or an got `metatensor.TensorMap` instance, got "
            f"{type(selected_samples)} instead"
        )

    if selected_properties is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_properties, Labels):
        selected_properties = selected_properties._as_mts_labels_t()
        c_options.selected_properties.subset = ctypes.pointer(selected_properties)
        c_options.__keepalive["selected_properties"] = selected_properties
    elif isinstance(selected_properties, TensorMap):
        c_options.selected_properties.predefined = selected_properties._ptr
    else:
        raise ValueError(
            "expected selected properties to be either an `metatensor.Labels` "
            "instance, or an got `metatensor.TensorMap` instance, got "
            f"{type(selected_properties)} instead"
        )

    if selected_keys is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_keys, Labels):
        selected_keys = selected_keys._as_mts_labels_t()
        c_options.selected_keys = ctypes.pointer(selected_keys)
        c_options.__keepalive["selected_keys"] = selected_keys
    return c_options


class CalculatorBase:
    """
    This is the base class for all calculators in featomic, providing the
    :py:meth:`CalculatorBase.compute` function.

    One can initialize a ``Calculator`` in two ways: either directly with the registered
    name and JSON parameter string (which are documented in the
    :ref:`userdoc-calculators`); or through one of the child class documented below.

    :param name: name used to register this calculator
    :param parameters: JSON parameter string for the calculator
    """

    def __init__(self, name: str, parameters: str):
        self._c_name = name
        self._lib = _get_library()
        self._as_parameter_ = self._lib.featomic_calculator(
            name.encode("utf8"), parameters.encode("utf8")
        )
        _check_featomic_pointer(self._as_parameter_)

        self._selected_samples = None
        self._selected_features = None

    def __del__(self):
        if hasattr(self, "_lib"):
            # if we failed to load the lib, don't double error by trying to call
            # ``self._lib.featomic_calculator_free``
            self._lib.featomic_calculator_free(self)
        self._as_parameter_ = 0

    @property
    def name(self) -> str:
        """name of this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.featomic_calculator_name(
                self, buffer, bufflen
            )
        )

    @property
    def c_name(self) -> str:
        """name used to register & create this calculator"""
        return self._c_name

    @property
    def parameters(self):
        """parameters (formatted as JSON) used to create this calculator"""
        return _call_with_growing_buffer(
            lambda buffer, bufflen: self._lib.featomic_calculator_parameters(
                self, buffer, bufflen
            )
        )

    @property
    def cutoffs(self) -> List[float]:
        """all the radial cutoffs used by this calculator's neighbors lists"""

        cutoffs = ctypes.POINTER(ctypes.c_double)()
        cutoffs_count = c_uintptr_t()
        self._lib.featomic_calculator_cutoffs(self, cutoffs, cutoffs_count)

        result = []
        for i in range(cutoffs_count.value):
            result.append(cutoffs[i])

        return result

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        *,
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
        selected_samples: Optional[Union[Labels, TensorMap]] = None,
        selected_properties: Optional[Union[Labels, TensorMap]] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        r"""Runs a calculation with this calculator on the given ``systems``.

        :param systems: single system or list of systems on which to run the
            calculation. The systems will automatically be wrapped into compatible
            classes (using :py:func:`featomic.systems.wrap_system`). Multiple types of
            systems are supported, see the documentation of
            :py:class:`featomic.IntoSystem` to get the full list.

        :param use_native_system: If ``True`` (this is the default), copy data from the
            ``systems`` into Rust ``SimpleSystem``. This can be a lot faster than having
            to cross the FFI boundary often when accessing the neighbor list. Otherwise
            the Python neighbor list is used.

        :param gradients: List of gradients to compute. If this is ``None`` or an empty
            list ``[]``, no gradients are computed. Gradients are stored inside the
            different blocks, and can be accessed with
            ``descriptor.block(...).gradient(<parameter>)``, where ``<parameter>`` is a
            string describing the gradients. The following gradients are available:

            - ``"positions"``, for gradients of the representation with respect to
              atomic positions, with fixed cell matrix parameters. Positions gradients
              are computed as

              .. math::
                  \frac{\partial \langle q \vert A_i \rangle}
                       {\partial \mathbf{r_j}}

              where :math:`\langle q \vert A_i \rangle` is the representation around
              atom :math:`i` and :math:`\mathbf{r_j}` is the position vector of the atom
              :math:`j`.

              **Note**: Position gradients of an atom are computed with respect to all
              other atoms within the representation. To recover the force one has to
              accumulate all pairs associated with atom :math:`i`.

            - ``"strain"``, for gradients of the representation with respect to strain.
              These gradients are typically used to compute the virial, and from there
              the pressure acting on a system. To compute them, we pretend that all the
              positions :math:`\mathbf r` and unit cell :math:`\mathbf H` have been
              scaled by a strain matrix :math:`\epsilon`:

              .. math::
                 \mathbf r &\rightarrow \mathbf r \left(\mathbb{1} + \epsilon \right)\\
                 \mathbf H &\rightarrow \mathbf H \left(\mathbb{1} + \epsilon \right)

              and then take the gradients of the representation with respect to this
              matrix:

              .. math::
                  \frac{\partial \langle q \vert A_i \rangle}
                       {\partial \mathbf{\epsilon}}

            - ``"cell"``, for gradients of the representation with respect to the
              system's cell parameters. These gradients are computed at fixed positions,
              and often not what you want when computing gradients explicitly (they are
              mainly used in ``featomic.torch`` to integrate with backward
              propagation). If you are trying to compute the virial or the stress, you
              should use ``"strain"`` gradients instead.

              .. math::
                  \left. \frac{\partial \langle q \vert A_i \rangle}
                       {\partial \mathbf{H}} \right |_\mathbf{r}

        :param selected_samples: Set of samples on which to run the calculation. Use
            ``None`` to run the calculation on all samples in the ``systems`` (this is
            the default).

            If ``selected_samples`` is an :py:class:`metatensor.TensorMap`, then the
            samples for each key will be used as-is when computing the representation.

            If ``selected_samples`` is a set of :py:class:`metatensor.Labels` containing
            the same variables as the default set of samples for this calculator, then
            only entries from the full set that also appear in ``selected_samples`` will
            be used.

            If ``selected_samples`` is a set of :py:class:`metatensor.Labels` containing
            a subset of the variables of the default set of samples, then only samples
            from the default set with the same values for these variables as one of the
            entries in ``selected_samples`` will be used.

        :param selected_properties: Set of properties to compute. Use ``None`` to run
            the calculation on all properties (this is the default).

            If ``selected_properties`` is an :py:class:`metatensor.TensorMap`, then the
            properties for each key will be used as-is when computing the
            representation.

            If ``selected_properties`` is a set of :py:class:`metatensor.Labels`
            containing the same variables as the default set of properties for this
            calculator, then only entries from the full set that also appear in
            ``selected_properties`` will be used.

            If ``selected_properties`` is a set of :py:class:`metatensor.Labels`
            containing a subset of the variables of the default set of properties, then
            only properties from the default set with the same values for these
            variables as one of the entries in ``selected_properties`` will be used.

        :param selected_keys: Selection for the keys to include in the output. If this
            is ``None``, the default set of keys (as determined by the calculator) will
            be used. Note that this default set of keys can depend on which systems we
            are running the calculation on.
        """

        c_systems = _convert_systems(systems)
        tensor_map_ptr = ctypes.POINTER(mts_tensormap_t)()

        c_options = _options_to_c(
            gradients=gradients,
            use_native_system=use_native_system,
            selected_samples=selected_samples,
            selected_properties=selected_properties,
            selected_keys=selected_keys,
        )
        self._lib.featomic_calculator_compute(
            self, tensor_map_ptr, c_systems, c_systems._length_, c_options
        )

        return TensorMap._from_ptr(tensor_map_ptr)
