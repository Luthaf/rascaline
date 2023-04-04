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

    c_options = rascal_calculation_options_t()
    c_options.gradients = c_gradients
    c_options.gradients_count = c_gradients._length_
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

    if selected_keys is None:
        # nothing to do, all pointers are already NULL
        pass
    elif isinstance(selected_keys, Labels):
        selected_keys = selected_keys._as_eqs_labels_t()
        c_options.selected_keys = ctypes.pointer(selected_keys)
        c_options.__keepalive["selected_keys"] = selected_keys
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
        *,
        gradients: Optional[List[str]] = None,
        use_native_system: bool = True,
        selected_samples: Optional[Union[Labels, TensorMap]] = None,
        selected_properties: Optional[Union[Labels, TensorMap]] = None,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        r"""Runs a calculation with this calculator on the given ``systems``.

        :param systems: single system or list of systems on which to run the
            calculation. The systems will automatically be wrapped into
            compatible classes (using :py:func:`rascaline.systems.wrap_system`).
            Multiple types of systems are supported, see the documentation of
            :py:class:`rascaline.IntoSystem` to get the full list.

        :param use_native_system: If ``True`` (this is the default), copy data
            from the ``systems`` into Rust ``SimpleSystem``. This can be a lot
            faster than having to cross the FFI boundary often when accessing
            the neighbor list. Otherwise the Python neighbor list is used.

        :param gradients: List of gradients to compute. If this is ``None`` or
            an empty list ``[]``, no gradients are computed. Gradients are
            stored inside the different blocks, and can be accessed with
            ``descriptor.block(...).gradient(<parameter>)``, where
            ``<parameter>`` is ``"positions"`` or ``"cell"``. The following
            gradients are available:

            - ``"positions"``, for gradients of the representation with respect to
              atomic positions. Positions gradients are computed as

              .. math::
                  \frac{\partial \langle q \vert A_i \rangle}
                       {\partial \mathbf{r_j}}

              where :math:`\langle q \vert A_i \rangle` is the representation around
              atom :math:`i` and :math:`\mathbf{r_j}` is the position vector of the
              atom :math:`j`.

              **Note**: Position gradients of an atom are computed with respect to all
              other atoms within the representation. To recover the force one has to
              accumulate all pairs associated with atom :math:`i`.

            - ``"cell"``, for gradients of the representation with respect to cell
              vectors. Cell gradients are computed as

              .. math::
                  \frac{\partial \langle q \vert A_i \rangle}
                       {\partial \mathbf{h}}

              where :math:`\mathbf{h}` is the cell matrix.

              **Note**: When computing the virial, one often needs to evaluate
              the gradient of the representation with respect to the strain
              :math:`\epsilon`. To recover the typical expression from the cell
              gradient one has to multiply the cell gradients with the cell
              matrix :math:`\mathbf{h}`

              .. math::
                  -\frac{\partial \langle q \vert A \rangle}
                        {\partial\epsilon}
                   = -\frac{\partial \langle q \vert A \rangle}
                           {\partial \mathbf{h}} \cdot \mathbf{h}

        :param selected_samples: Set of samples on which to run the calculation.
            Use ``None`` to run the calculation on all samples in the
            ``systems`` (this is the default).

            If ``selected_samples`` is an :py:class:`equistore.TensorMap`, then
            the samples for each key will be used as-is when computing the
            representation.

            If ``selected_samples`` is a set of :py:class:`equistore.Labels`
            containing the same variables as the default set of samples for this
            calculator, then only entries from the full set that also appear in
            ``selected_samples`` will be used.

            If ``selected_samples`` is a set of :py:class:`equistore.Labels`
            containing a subset of the variables of the default set of samples,
            then only samples from the default set with the same values for
            these variables as one of the entries in ``selected_samples`` will
            be used.

        :param selected_properties: Set of properties to compute. Use ``None``
            to run the calculation on all properties (this is the default).

            If ``selected_properties`` is an :py:class:`equistore.TensorMap`,
            then the properties for each key will be used as-is when computing
            the representation.

            If ``selected_properties`` is a set of :py:class:`equistore.Labels`
            containing the same variables as the default set of properties for
            this calculator, then only entries from the full set that also
            appear in ``selected_properties`` will be used.

            If ``selected_properties`` is a set of :py:class:`equistore.Labels`
            containing a subset of the variables of the default set of
            properties, then only properties from the default set with the same
            values for these variables as one of the entries in
            ``selected_properties`` will be used.

        :param selected_keys: Selection for the keys to include in the output.
            If this is ``None``, the default set of keys (as determined by the
            calculator) will be used. Note that this default set of keys can
            depend on which systems we are running the calculation on.
        """

        c_systems = _convert_systems(systems)
        tensor_map_ptr = ctypes.POINTER(eqs_tensormap_t)()

        c_options = _options_to_c(
            gradients=gradients,
            use_native_system=use_native_system,
            selected_samples=selected_samples,
            selected_properties=selected_properties,
            selected_keys=selected_keys,
        )
        self._lib.rascal_calculator_compute(
            self, tensor_map_ptr, c_systems, c_systems._length_, c_options
        )

        return TensorMap._from_ptr(tensor_map_ptr)


class AtomicComposition(CalculatorBase):
    """An atomic composition calculator for obtaining the stoichiometric information.

    For ``per_structure=False`` calculator has one property ``count`` that is
    ``1`` for all centers, and has a sample index that indicates the central atom type.

    For ``per_structure=True`` a sum for each structure is performed. The number of
    atoms per structure is saved. The only sample left is names ``structure``.
    """

    def __init__(self, per_structure):
        parameters = {
            "per_structure": per_structure,
        }
        super().__init__("atomic_composition", parameters)


class DummyCalculator(CalculatorBase):
    def __init__(self, cutoff, delta, name):
        parameters = {
            "cutoff": cutoff,
            "delta": delta,
            "name": name,
        }
        super().__init__("dummy_calculator", parameters)


class NeighborList(CalculatorBase):
    """
    This calculator computes the neighbor list for a given spherical cutoff, and
    returns the list of distance vectors between all pairs of atoms strictly
    inside the cutoff.

    Users can request either a "full" neighbor list (including an entry for both
    ``i - j`` pairs and ``j - i`` pairs) or save memory/computational by only
    working with "half" neighbor list (only including one entry for each ``i/j``
    pair)

    Self pairs (pairs between an atom and periodic copy itself) can appear when
    the cutoff is larger than the cell under periodic boundary conditions. Self
    pairs with a distance of 0 are not included in this calculator, even though
    they are required when computing SOAP.

    This sample produces a single property (``"distance"``) with three
    components (``"pair_direction"``) containing the x, y, and z component of
    the vector from the first atom in the pair to the second. In addition to the
    atom indexes, the samples also contain a pair index, to be able to
    distinguish between multiple pairs between the same atom (if the cutoff is
    larger than the cell).
    """

    def __init__(self, cutoff, full_neighbor_list):
        parameters = {
            "cutoff": cutoff,
            "full_neighbor_list": full_neighbor_list,
        }
        super().__init__("neighbor_list", parameters)


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

    The spherical expansion is at the core of representations in the SOAP family
    of descriptors. The spherical expansion represent atomic density as a
    collection of Gaussian functions centered on each atom, and then represent
    the local density around each atom (up to a cutoff) on a basis of radial
    functions and spherical harmonics. This representation is not rotationally
    invariant, for that you should use the :py:class:`SoapPowerSpectrum` class.

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
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("spherical_expansion", parameters)


class SphericalExpansionByPair(CalculatorBase):
    """
    Pair-by-pair version of the spherical expansion of Smooth Overlap of Atomic
    Positions (SOAP).

    This is usually an intermediary step when computing the full
    :py:class:`SphericalExpansion`, which most users do not need to care about.
    When computing SOAP, the density around the central atom :math:`i` is a sum
    of pair contributions:

    .. math::
        \\rho_i(\\mathbf{r}) = \\sum_j g(\\mathbf{r_{ji}} - \\mathbf{r}).

    The full spherical expansion is then computed as a sum over all pairs within
    the cutoff:

    .. math::
        \\int Y^l_m(\\mathbf{r})\\ R_n(\\mathbf{r}) \\, \\rho_i(\\mathbf{r})
            \\mathrm{d}\\mathbf{r} = \\sum_j C^{ij}_{nlm}

        C^{ij}_{nlm} = \\int Y^l_m(\\mathbf{r}) \\ R_n(\\mathbf{r}) \\,
            g(\\mathbf{r_{ji}} - \\mathbf{r}) \\, \\mathrm{d}\\mathbf{r}

    This calculators computes the individual :math:`C^{ij}_{nlm}` terms, which
    can then be used to build multi-center representations such as the ones
    discussed in `"Unified theory of atom-centered representations and
    message-passing machine-learning schemes"
    <https://doi.org/10.1063/5.0087042>`_.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <spherical-expansion-by-pair>`.
    """

    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        radial_basis,
        center_atom_weight,
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
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("spherical_expansion_by_pair", parameters)


class SoapRadialSpectrum(CalculatorBase):
    """Radial spectrum of Smooth Overlap of Atomic Positions (SOAP).

    The SOAP radial spectrum represent each atom by the radial average of the
    density of its neighbors. It is very similar to a radial distribution
    function `g(r)`. It is a 2-body representation, only containing information
    about the distances between atoms.

    See `this review article <https://doi.org/10.1063/1.5090481>`_ for more
    information on the SOAP representation, and `this paper
    <https://doi.org/10.1063/5.0044689>`_ for information on how it is
    implemented in rascaline.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <soap-radial-spectrum>`.
    """

    def __init__(
        self,
        cutoff,
        max_radial,
        atomic_gaussian_width,
        center_atom_weight,
        radial_basis,
        cutoff_function,
        radial_scaling=None,
    ):
        parameters = {
            "cutoff": cutoff,
            "max_radial": max_radial,
            "atomic_gaussian_width": atomic_gaussian_width,
            "center_atom_weight": center_atom_weight,
            "radial_basis": radial_basis,
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("soap_radial_spectrum", parameters)


class SoapPowerSpectrum(CalculatorBase):
    """Power spectrum of Smooth Overlap of Atomic Positions (SOAP).

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
            "cutoff_function": cutoff_function,
        }

        if radial_scaling is not None:
            parameters["radial_scaling"] = radial_scaling

        super().__init__("soap_power_spectrum", parameters)


class LodeSphericalExpansion(CalculatorBase):
    """Long-Distance Equivariant (LODE).

    The spherical expansion is at the core of representations in the LODE
    family. The LODE spherical
    expansion represent atomic density as a collection of 'decorated' gaussian
    functions centered on each atom, and then represent the local density around
    each atom on a basis of radial functions and spherical harmonics.
    This representation is not rotationally invariant.

    See [this article](https://aip.scitation.org/doi/10.1063/1.5128375)
    for more information on the LODE representation.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <lode-spherical-expansion>`.
    """

    def __init__(
        self,
        cutoff,
        max_radial,
        max_angular,
        atomic_gaussian_width,
        center_atom_weight,
        potential_exponent,
        radial_basis,
        k_cutoff=None,
    ):
        parameters = {
            "cutoff": cutoff,
            "k_cutoff": k_cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "atomic_gaussian_width": atomic_gaussian_width,
            "center_atom_weight": center_atom_weight,
            "potential_exponent": potential_exponent,
            "radial_basis": radial_basis,
        }

        super().__init__("lode_spherical_expansion", parameters)
