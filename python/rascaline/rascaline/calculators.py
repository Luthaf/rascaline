import json


try:
    # see rascaline-torch/calculators.py for the explanation of what's going on here
    _ = CalculatorBase
except NameError:
    from .calculator_base import CalculatorBase


class AtomicComposition(CalculatorBase):
    """An atomic composition calculator for obtaining the stoichiometric information.

    For ``per_system=False`` calculator has one property ``count`` that is ``1`` for all
    atoms, and has a sample index that indicates the central atom type.

    For ``per_system=True`` a sum for each system is performed. The number of atoms per
    system is saved. The only sample left is named ``system``.
    """

    def __init__(self, *, per_system):
        parameters = {
            "per_system": per_system,
        }
        super().__init__("atomic_composition", json.dumps(parameters))


class DummyCalculator(CalculatorBase):
    def __init__(self, *, cutoff, delta, name):
        parameters = {
            "cutoff": cutoff,
            "delta": delta,
            "name": name,
        }
        super().__init__("dummy_calculator", json.dumps(parameters))


class NeighborList(CalculatorBase):
    """
    This calculator computes the neighbor list for a given spherical cutoff, and returns
    the list of distance vectors between all pairs of atoms strictly inside the cutoff.

    Users can request either a "full" neighbor list (including an entry for both ``i -
    j`` pairs and ``j - i`` pairs) or save memory/computational by only working with
    "half" neighbor list (only including one entry for each ``i/j`` pair)

    Pairs between an atom and it's own periodic copy can appear when the cutoff is
    larger than the cell under periodic boundary conditions. Self pairs with a distance
    of 0 (i.e. self pairs inside the original unit cell) are only included when using
    ``self_pairs=True``.

    This calculator produces a single property (``"distance"``) with three components
    (``"pair_xyz"``) containing the x, y, and z component of the distance vector of the
    pair.

    The samples contain the two atoms indexes, as well as the number of cell boundaries
    crossed to create this pair.
    """

    def __init__(
        self,
        *,
        cutoff: float,
        full_neighbor_list: bool,
        self_pairs: bool = False,
    ):
        parameters = {
            "cutoff": cutoff,
            "full_neighbor_list": full_neighbor_list,
            "self_pairs": self_pairs,
        }
        super().__init__("neighbor_list", json.dumps(parameters))


class SortedDistances(CalculatorBase):
    """Sorted distances vector representation of an atomic environment.

    Each atomic center is represented by a vector of distance to its neighbors
    within the spherical ``cutoff``, sorted from smallest to largest. If there
    are less neighbors than ``max_neighbors``, the remaining entries are filled
    with ``cutoff`` instead.

    Separate atomic types for neighbors are represented separately, meaning that the
    ``max_neighbors`` parameter only apply to a single atomic type.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <sorted-distances>`.
    """

    def __init__(self, *, cutoff, max_neighbors, separate_neighbor_types):
        parameters = {
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
            "separate_neighbor_types": separate_neighbor_types,
        }
        super().__init__("sorted_distances", json.dumps(parameters))


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

    def __init__(self, *, cutoff, density, basis, **kwargs):
        if len(kwargs) != 0:
            raise ValueError("TODO: old style parameters")

        parameters = {
            "cutoff": cutoff,
            "density": density,
            "basis": basis,
        }

        super().__init__("spherical_expansion", json.dumps(parameters))


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

    def __init__(self, *, cutoff, density, basis, **kwargs):
        if len(kwargs) != 0:
            raise ValueError("TODO: old style parameters")

        parameters = {
            "cutoff": cutoff,
            "density": density,
            "basis": basis,
        }

        super().__init__("spherical_expansion_by_pair", json.dumps(parameters))


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

    def __init__(self, *, cutoff, density, basis, **kwargs):
        if len(kwargs) != 0:
            raise ValueError("TODO: old style parameters")

        parameters = {
            "cutoff": cutoff,
            "density": density,
            "basis": basis,
        }

        super().__init__("soap_radial_spectrum", json.dumps(parameters))


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

    .. seealso::
        :py:class:`rascaline.utils.PowerSpectrum` is an implementation that
        allows to compute the power spectrum from different spherical expansions.
    """

    def __init__(self, *, cutoff, density, basis, **kwargs):
        if len(kwargs) != 0:
            raise ValueError("TODO: old style parameters")

        parameters = {
            "cutoff": cutoff,
            "density": density,
            "basis": basis,
        }

        super().__init__("soap_power_spectrum", json.dumps(parameters))


class LodeSphericalExpansion(CalculatorBase):
    """Long-Distance Equivariant (LODE).

    The spherical expansion is at the core of representations in the LODE
    family. The LODE spherical
    expansion represent atomic density as a collection of 'decorated' gaussian
    functions centered on each atom, and then represent the local density around
    each atom on a basis of radial functions and spherical harmonics.
    This representation is not rotationally invariant.

    See `this article <https://aip.scitation.org/doi/10.1063/1.5128375>`_
    for more information on the LODE representation.

    For a full description of the hyper-parameters, see the corresponding
    :ref:`documentation <lode-spherical-expansion>`.
    """

    def __init__(self, *, density, basis, k_cutoff=None, **kwargs):
        if len(kwargs) != 0:
            raise ValueError("TODO: old style parameters")

        parameters = {
            "k_cutoff": k_cutoff,
            "density": density,
            "basis": basis,
        }

        super().__init__("lode_spherical_expansion", json.dumps(parameters))
