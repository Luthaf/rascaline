import json


try:
    # see rascaline-torch/calculators.py for the explanation of what's going on here
    _ = CalculatorBase
except NameError:
    from .calculator_base import CalculatorBase


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
        super().__init__("atomic_composition", json.dumps(parameters))


class DummyCalculator(CalculatorBase):
    def __init__(self, cutoff, delta, name):
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

    The ``quantity`` parameter determine what will be included in the output. It can
    take one of three values:

    - ``"Distance"``, to get the distance between the atoms, accounting for periodic
      boundary conditions. This is the default.
    - ``"CellShiftVector"``, to get the cell shift vector, which can then be used to
      apply periodic boundary conditions and compute the distance vector.

      If ``S`` is the cell shift vector, ``rij`` the pair distance vector, ``ri`` and
      ``rj`` the position of the atoms, ``rij = rj - ri + S``.
    - ``"CellShiftIndices"``, to get three integers indicating the number of cell
      vectors (``A``, ``B``, and ``C``) entering the cell shift.

      If the cell vectors are ``A``, ``B``, and ``C``, this returns three integers
      ``sa``, ``sb``, and ``sc`` such that the cell shift ``S = sa * A + sb * B + sc *
      C``.

    This calculator produces a single property (``"distance"``, ``"cell_shift_vector"``,
    or ``"cell_shift_indices"``) with three components (``"pair_direction"``) containing
    the x, y, and z component of the requested vector. In addition to the atom indexes,
    the samples also contain a pair index, to be able to distinguish between multiple
    pairs between the same atom (if the cutoff is larger than the cell).
    """

    def __init__(
        self,
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

        super().__init__("lode_spherical_expansion", json.dumps(parameters))
