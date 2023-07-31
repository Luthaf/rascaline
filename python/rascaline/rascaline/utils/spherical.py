"""
Classes and functions to aid converting to and from, and operating within, the
spherical basis. Contains classes WignerDReal and ClebschGordanReal, as well as
functions to perform Clebsch-Gordan iterations. Code mostly taken from
librascal:

github.com/lab-cosmo/librascal/blob/master/bindings/rascal/utils/cg_utils.py
"""
from copy import deepcopy
from itertools import product
import re
from typing import Optional, Tuple, Sequence

import ase
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import wigners

import equistore
from equistore import Labels, TensorBlock, TensorMap
import rascaline


# ===== WignerDReal class for describing rotations


class WignerDReal:
    """
    A helper class to compute Wigner D matrices given the Euler angles of a rotation,
    and apply them to spherical harmonics (or coefficients). Built to function with
    real-valued coefficients.
    """

    def __init__(self, lmax, alpha, beta, gamma):
        """
        Initialize the WignerDReal class.
        lmax: int
            maximum angular momentum channel for which the Wigner D matrices are
            computed
        alpha, beta, gamma: float
            Euler angles, in radians
        """
        self._lmax = lmax

        self._rotation = cartesian_rotation(alpha, beta, gamma)

        r2c_mats = {}
        c2r_mats = {}
        for L in range(0, self._lmax + 1):
            r2c_mats[L] = np.hstack(
                [_r2c(np.eye(2 * L + 1)[i])[:, np.newaxis] for i in range(2 * L + 1)]
            )
            c2r_mats[L] = np.conjugate(r2c_mats[L]).T
        self._wddict = {}
        for L in range(0, self._lmax + 1):
            wig = _wigner_d(L, alpha, beta, gamma)
            self._wddict[L] = np.real(c2r_mats[L] @ np.conjugate(wig) @ r2c_mats[L])

    def rotate(self, rho):
        """
        Rotates a vector of 2l+1 spherical harmonics (coefficients) according to the
        rotation defined in the initialization.
        rho: array
            List of 2l+1 coefficients
        Returns:
        --------
        (2l+1) array containing the coefficients for the rotated structure
        """

        L = (rho.shape[-1] - 1) // 2
        return rho @ self._wddict[L].T

    def rotate_frame(self, frame, in_place=False):
        """
        Utility function to also rotate a structure, given as an Atoms frame.
        NB: it will rotate positions and cell, and no other array.
        frame: ase.Atoms
            An atomic structure in ASE format, that will be modified in place
        in_frame: bool
            Whether the frame should be copied or processed in place (defaults to False)
        Returns:
        -------
        The rotated frame.
        """

        if is_ase_Atoms(frame):
            if in_place:
                frame = frame.copy()
            frame.positions = frame.positions @ self._rotation.T
            frame.cell = frame.cell @ self._rotation.T
        else:
            if in_place:
                frame = deepcopy(frame)
            frame["positions"] = self._rotation @ frame["positions"]
            frame["cell"] = self._rotation @ frame["cell"]
        return frame

    def rotate_coeff_vector(
        self,
        frame: ase.Atoms,
        coeffs: np.ndarray,
        lmax: dict,
        nmax: dict,
    ) -> np.ndarray:
        """
        Rotates the irreducible spherical components (ISCs) of basis set
        coefficients in the spherical basis passed in as a flat vector.

        Given the basis set definition specified by ``lmax`` and ``nmax``, the
        assumed ordering of basis function coefficients follows the following
        hierarchy, which can be read as nested loops over the various indices.
        Be mindful that some indices range are from 0 to x (exclusive) and
        others from 0 to x + 1 (exclusive). The ranges reported below are
        ordered.

        1. Loop over atoms (index ``i``, of chemical species ``a``) in the
        structure. ``i`` takes values 0 to N (** exclusive **), where N is the
        number of atoms in the structure.

        2. Loop over spherical harmonics channel (index ``l``) for each atom.
        ``l`` takes values from 0 to ``lmax[a] + 1`` (** exclusive **), where
        ``a`` is the chemical species of atom ``i``, given by the chemical
        symbol at the ``i``th position of ``symbol_list``.

        3. Loop over radial channel (index ``n``) for each atom ``i`` and
        spherical harmonics channel ``l`` combination. ``n`` takes values from 0
        to ``nmax[(a, l)]`` (** exclusive **).

        4. Loop over spherical harmonics component (index ``m``) for each atom.
        ``m`` takes values from ``-l`` to ``l`` (** inclusive **).

        :param frame: the atomic structure in ASE format for which the
            coefficients are defined.
        :param coeffs: the coefficients in the spherical basis, as a flat
            vector.
        :param lmax: dict containing the maximum spherical harmonics (l) value
            for each atom type.
        :param nmax: dict containing the maximum radial channel (n) value for
            each combination of atom type and l.

        :return: the rotated coefficients in the spherical basis, as a flat
            vector with the same order as the input vector.
        """
        # Initialize empty vector for storing the rotated ISCs
        rot_vect = np.empty_like(coeffs)

        # Iterate over atomic species of the atoms in the frame
        curr_idx = 0
        for symbol in frame.get_chemical_symbols():
            # Get the basis set lmax value for this species
            sym_lmax = lmax[symbol]
            for l in range(sym_lmax + 1):
                # Get the number of radial functions for this species and l value
                sym_l_nmax = nmax[(symbol, l)]
                # Get the Wigner D Matrix for this l value
                wig_mat = self._wddict[l].T
                for n in range(sym_l_nmax):
                    # Retrieve the irreducible spherical component
                    isc = coeffs[curr_idx : curr_idx + (2 * l + 1)]
                    # Rotate the ISC and store
                    rot_isc = isc @ wig_mat
                    rot_vect[curr_idx : curr_idx + (2 * l + 1)][:] = rot_isc[:]
                    # Update the start index for the next ISC
                    curr_idx += 2 * l + 1

        return rot_vect

    def rotate_tensorblock(self, l: int, block: TensorBlock) -> TensorBlock:
        """
        Rotates a TensorBlock ``block``, represented in the spherical basis,
        according to the Wigner D Real matrices for the given ``l`` value.
        Assumes the components of the block are [("spherical_harmonics_m",),].
        """

        # Get the Wigner matrix for this l value
        wig = self._wddict[l].T

        # Copy the block
        block_rotated = block.copy()
        vals = block_rotated.values

        # Perform the rotation, either with numpy or torch, by taking the
        # tensordot product of the irreducible spherical components. Modify in-place the
        # values of the copied TensorBlock
        if isinstance(vals, torch.Tensor):
            wig = torch.tensor(wig)
            block_rotated.values[:] = torch.tensordot(
                vals.swapaxes(1, 2), wig, dims=1
            ).swapaxes(1, 2)
        elif isinstance(block.values, np.ndarray):
            block_rotated.values[:] = np.tensordot(
                vals.swapaxes(1, 2), wig, axes=1
            ).swapaxes(1, 2)
        else:
            raise TypeError("TensorBlock values must be a numpy array or torch tensor.")

        return block_rotated

    def rotate_tensormap(self, tensor: TensorMap) -> TensorMap:
        """
        Rotates a TensorMap usign Wigner D Matrices. Assumes the tensor keys has
        a name "spherical_harmonics_l" that indicates the l value, and that each
        block has exactly one component axis, named by
        ("spherical_harmonics_m",).
        """
        # Retrieve the key and the position of the l value in the key names
        keys = tensor.keys
        idx_l_value = keys.names.index("spherical_harmonics_l")

        # Iterate over the blocks and rotate
        rotated_blocks = []
        for key in keys:
            # Retrieve the l value
            l = key[idx_l_value]

            # Rotate the block and store
            rotated_blocks.append(self.rotate_tensorblock(l, tensor[key]))

        return TensorMap(keys, rotated_blocks)


# ===== helper functions for WignerDReal


def rotate_ase_frame(frame) -> Tuple[ase.Atoms, Tuple[float, float, float]]:
    """
    Make a copy of the input ``frame``. Randomly rotates its xyz and cell
    coordinates and returns the new frame, and euler angles alpha, beta, and
    gamma.
    """
    # Randomly generate euler angles between 0 and pi
    alpha, beta, gamma = np.random.uniform(size=(3)) * np.pi
    # Build cartesian rotation matrix
    R = cartesian_rotation(alpha, beta, gamma)
    # Copy the frame
    rotated_frame = frame.copy()
    # Rotate its positions and cell
    rotated_frame.positions = rotated_frame.positions @ R.T
    rotated_frame.cell = rotated_frame.cell @ R.T

    return rotated_frame, (alpha, beta, gamma)


def _wigner_d(l, alpha, beta, gamma):
    """Computes a Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    return np.complex128(wigner_d(l, alpha, beta, gamma))


def cartesian_rotation(alpha, beta, gamma):
    """A Cartesian rotation matrix in the appropriate convention
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()


def is_ase_Atoms(frame):
    is_ase = True
    if not hasattr(frame, "get_cell"):
        is_ase = False
    if not hasattr(frame, "get_positions"):
        is_ase = False
    if not hasattr(frame, "get_atomic_numbers"):
        is_ase = False
    if not hasattr(frame, "get_pbc"):
        is_ase = False
    return


def check_equivariance(
    unrotated: TensorMap,
    rotated: TensorMap,
    lmax: int,
    alpha: float,
    beta: float,
    gamma: float,
    rtol: Optional[float] = 1e-15,
    atol: Optional[float] = 1e-15,
    n_checks_per_block: Optional[int] = None,
) -> bool:
    """
    Checks equivariance by comparing the expansion coefficients of the
    structural representations of an unrotated and rotated structure, rotating
    the component vectors of the unrotated structure using a Wigner D-Matrix
    constructed using parameters ``lmax``, ``alpha``, ``beta``, ``gamma``.

    If ``n_checks_per_block`` is passed (i.e. not None, the default), only this
    number of (sample, property) combinations are checked per block. Otherwise,
    all component vectors in every block are checked.

    :param unrotated: a TensorMap of the coefficients in the spherical basis for
        the unrotated structure.
    :param rotated: a TensorMap of the coefficients in the spherical basis for
        the rotated structure.
    :param lmax: the maximum l value for which the spherical basis is expanded.
    :param alpha: the first Euler angle for the rotation between the unrotated
        and rotated structure.
    :param beta: the second Euler angle for the rotation between the unrotated
        and rotated structure.
    :param gamma: the third Euler angle for the rotation between the unrotated
        and rotated structure.
    :param rtol: the relative tolerance for the check. Default 1e-15.
    :param atol: the absolute tolerance for the check. Default 1e-15.
    :param n_checks_per_block: the number of comparisons between rotated and
        unrotated structures to perform per block of the input TensorMaps.

    :return bool: True if the rotated and unrotated structures are exact rotated
        forms of eachother in the spherical basis, within the defined
        tolerances. False otherwise.
    """
    equivariant = True

    # Check that the metadata is equivalent
    equistore.equal_metadata(unrotated, rotated)

    # Build Wigner D-Matrices
    wigner_d_matrices = WignerDReal(lmax, alpha, beta, gamma)

    # Check each block in turn
    for key in rotated.keys:
        # Access the blocks to compare
        unr_block = unrotated[key]
        rot_block = rotated[key]

        # Define the number of samples and properties
        n_samples = len(unr_block.samples)
        n_props = len(unr_block.properties)

        # If ``n_checks_per_block`` is passed, define a subset of samples and
        # properties to check
        samps_props = list(product(range(n_samples), range(n_props)))
        samps_props = (
            samps_props
            if n_checks_per_block is None
            else samps_props[:n_checks_per_block]
        )
        for sample_i, property_i in samps_props:
            # Get the component vectors, each of length (2 \lambda + 1)
            try:
                unr_comp_vect = (
                    unr_block.values[sample_i, ..., property_i].detach().numpy()
                )
                rot_comp_vect = (
                    rot_block.values[sample_i, ..., property_i].detach().numpy()
                )
            except AttributeError:
                unr_comp_vect = unr_block.values[sample_i, ..., property_i]
                rot_comp_vect = rot_block.values[sample_i, ..., property_i]

            # Rotate the unrotated components vector with a wigner D-matrix
            unr_comp_vect_rot = wigner_d_matrices.rotate(unr_comp_vect)

            # Check for exact (within a strict tolerance) equivalence
            if not np.allclose(unr_comp_vect_rot, rot_comp_vect, rtol=rtol, atol=atol):
                print(
                    f"block {key}, sample {unr_block.samples[sample_i]},"
                    + f" property {unr_block.properties[property_i]}, vectors"
                    + f" not equivariant: {unr_comp_vect_rot}, {rot_comp_vect}"
                )
                equivariant = False
    return equivariant


# ===== CleschGordanReal class


class ClebschGordanReal:
    def __init__(self, l_max):
        self._l_max = l_max
        self._cg = {}

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for L in range(0, self._l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(self._l_max + 1):
            for l2 in range(self._l_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[L].T

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg

    def combine(self, rho1, rho2, L):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError("Requested CG entry has not been precomputed")

        n_items = rho1.shape[0]
        n_features = rho1.shape[2]
        if rho1.shape[0] != rho2.shape[0] or rho1.shape[2] != rho2.shape[2]:
            raise IndexError("Cannot combine differently-shaped feature blocks")

        rho = np.zeros((n_items, 2 * L + 1, n_features))
        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M] += rho1[:, m1, :] * rho2[:, m2, :] * cg

        return rho

    def combine_einsum(self, rho1, rho2, L, combination_string):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = rho1.shape[0]
        if rho1.shape[0] != rho2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        features = np.einsum(combination_string, rho1[:, 0, ...], rho2[:, 0, ...]).shape
        rho = np.zeros((n_items, 2 * L + 1) + features[1:])

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M, ...] += np.einsum(
                        combination_string, rho1[:, m1, ...], rho2[:, m2, ...] * cg
                    )

        return rho

    def couple(self, decoupled, iterate=0):
        """
        Goes from an uncoupled product basis to a coupled basis. A
        (2l1+1)x(2l2+1) matrix transforming like the outer product of Y^m1_l1
        Y^m2_l2 can be rewritten as a list of coupled vectors, each transforming
        like a Y^L irrep.

        The process can be iterated: a D dimensional array that is the product
        of D Y^m_l can be turned into a set of multiple terms transforming as a
        single Y^M_L.

        decoupled: array or dict
            (...)x(2l1+1)x(2l2+1) array containing coefficients that transform
            like products of Y^l1 and Y^l2 harmonics. can also be called on a
            array of higher dimensionality, in which case the result will
            contain matrices of entries. If the further index also correspond to
            spherical harmonics, the process can be iterated, and couple() can
            be called onto its output, in which case the decoupling is applied
            to each entry.

        iterate: int
            calls couple iteratively the given number of times. equivalent to
            couple(couple(... couple(decoupled)))

        Returns:
        --------
        A dictionary tracking the nature of the coupled objects. When called one
        time, it returns a dictionary containing (l1, l2) [the coefficients of
        the parent Ylm] which in turns is a dictionary of coupled terms, in the
        form L:(...)x(2L+1)x(...) array. When called multiple times, it applies
        the coupling to each term, and keeps track of the additional l terms, so
        that e.g. when called with iterate=1 the return dictionary contains
        terms of the form (l3,l4,l1,l2) : { L: array }


        Note that this coupling scheme is different from the NICE-coupling where
        angular momenta are coupled from left to right as (((l1 l2) l3) l4)... )
        Thus results may differ when combining more than two angular channels.
        """

        coupled = {}

        # when called on a matrix, turns it into a dict form to which we can
        # apply the generic algorithm
        if not isinstance(decoupled, dict):
            l2 = (decoupled.shape[-1] - 1) // 2
            decoupled = {(): {l2: decoupled}}

        # runs over the tuple of (partly) decoupled terms
        for ltuple, lcomponents in decoupled.items():
            # each is a list of L terms
            for lc in lcomponents.keys():
                # this is the actual matrix-valued coupled term,
                # of shape (..., 2l1+1, 2l2+1), transforming as Y^m1_l1 Y^m2_l2
                dec_term = lcomponents[lc]
                l1 = (dec_term.shape[-2] - 1) // 2
                l2 = (dec_term.shape[-1] - 1) // 2

                # there is a certain redundance: the L value is also the last entry
                # in ltuple
                if lc != l2:
                    raise ValueError(
                        "Inconsistent shape for coupled angular momentum block."
                    )

                # in the new coupled term, prepend (l1,l2) to the existing label
                coupled[(l1, l2) + ltuple] = {}
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    Lterm = np.zeros(shape=dec_term.shape[:-2] + (2 * L + 1,))
                    for M in range(2 * L + 1):
                        for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                            Lterm[..., M] += dec_term[..., m1, m2] * cg
                    coupled[(l1, l2) + ltuple][L] = Lterm

        # repeat if required
        if iterate > 0:
            coupled = self.couple(coupled, iterate - 1)
        return coupled

    def decouple(self, coupled, iterate=0):
        """
        Undoes the transformation enacted by couple.
        """

        decoupled = {}
        # applies the decoupling to each entry in the dictionary
        for ltuple, lcomponents in coupled.items():
            # the initial pair in the key indicates the decoupled terms that generated
            # the L entries
            l1, l2 = ltuple[:2]

            # shape of the coupled matrix (last entry is the 2L+1 M terms)
            shape = next(iter(lcomponents.values())).shape[:-1]

            dec_term = np.zeros(
                shape
                + (
                    2 * l1 + 1,
                    2 * l2 + 1,
                )
            )
            for L in range(max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1):
                # supports missing L components, e.g. if they are zero because of symmetry
                if not L in lcomponents:
                    continue
                for M in range(2 * L + 1):
                    for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                        dec_term[..., m1, m2] += cg * lcomponents[L][..., M]
            # stores the result with a key that drops the l's we have just decoupled
            if not ltuple[2:] in decoupled:
                decoupled[ltuple[2:]] = {}
            decoupled[ltuple[2:]][l2] = dec_term

        # rinse, repeat
        if iterate > 0:
            decoupled = self.decouple(decoupled, iterate - 1)

        # if we got a fully decoupled state, just return an array
        if ltuple[2:] == ():
            decoupled = next(iter(decoupled[()].values()))
        return decoupled


# ===== helper functions for ClebschGordanReal


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


I_SQRT_2 = 1.0 / np.sqrt(2)
SQRT_2 = np.sqrt(2)


def _r2c(sp):
    """Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l"""

    l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = np.zeros(len(sp), dtype=np.complex128)
    rc[l] = sp[l]
    for m in range(1, l + 1):
        rc[l + m] = (sp[l + m] + 1j * sp[l - m]) * I_SQRT_2 * (-1) ** m
        rc[l - m] = (sp[l + m] - 1j * sp[l - m]) * I_SQRT_2
    return rc


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)


# ======== Fxns used to perform CG iterations


def acdc_standardize_keys(descriptor):
    """
    Standardize the naming scheme of density expansion coefficient blocks
    (nu=1)
    """

    key_names = descriptor.keys.names
    if not "spherical_harmonics_l" in key_names:
        raise ValueError(
            "Descriptor missing spherical harmonics channel key `spherical_harmonics_l`"
        )
    blocks = []
    keys = []
    for key, block in descriptor.items():
        key = tuple(key)
        if not "inversion_sigma" in key_names:
            key = (1,) + key
        if not "order_nu" in key_names:
            key = (1,) + key
        keys.append(key)
        property_names = _remove_suffix(block.properties.names, "_1")
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(property_names, block.properties.values),
            )
        )

    if not "inversion_sigma" in key_names:
        key_names = ["inversion_sigma"] + key_names
    if not "order_nu" in key_names:
        key_names = ["order_nu"] + key_names

    return TensorMap(
        keys=Labels(names=key_names, values=np.asarray(keys, dtype=np.int32)),
        blocks=blocks,
    )


def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    other_keys_match=None,
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("inversion_sigma", "spherical_harmonics_l",
    "order_nu"). The automatically-determined naming of output features can be
    overridden by giving a list of "feature_names". By defaults, all other key
    labels are combined in an "outer product" mode, i.e. if there is a key-side
    neighbor_species in both x_a and x_b, the returned keys will have two
    neighbor_species labels, corresponding to the parent features. By providing
    a list `other_keys_match` of keys that should match, these are not
    outer-producted, but combined together. for instance, passing `["species
    center"]` means that the keys with the same species center will be combined
    together, but yield a single key with the same species_center in the
    results.
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    other_keys_a = tuple(
        name
        for name in x_a.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )
    other_keys_b = tuple(
        name
        for name in x_b.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )

    if other_keys_match is None:
        OTHER_KEYS = [k + "_a" for k in other_keys_a] + [k + "_b" for k in other_keys_b]
    else:
        OTHER_KEYS = (
            other_keys_match
            + [
                k + ("_a" if k in other_keys_b else "")
                for k in other_keys_a
                if k not in other_keys_match
            ]
            + [
                k + ("_b" if k in other_keys_a else "")
                for k in other_keys_b
                if k not in other_keys_match
            ]
        )

    # we assume grad components are all the same
    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).properties.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).properties.names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a.items():
        lam_a = index_a["spherical_harmonics_l"]
        sigma_a = index_a["inversion_sigma"]
        order_a = index_a["order_nu"]
        properties_a = (
            block_a.properties
        )  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples

        # and x_b
        for index_b, block_b in x_b.items():
            lam_b = index_b["spherical_harmonics_l"]
            sigma_b = index_b["inversion_sigma"]
            order_b = index_b["order_nu"]
            properties_b = block_b.properties
            samples_b = block_b.samples

            if other_keys_match is None:
                OTHERS = tuple(index_a[name] for name in other_keys_a) + tuple(
                    index_b[name] for name in other_keys_b
                )
            else:
                OTHERS = tuple(
                    index_a[k] for k in other_keys_match if index_a[k] == index_b[k]
                )
                # skip combinations without matching key
                if len(OTHERS) < len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                OTHERS = OTHERS + tuple(
                    index_a[k] for k in other_keys_a if k not in other_keys_match
                )
                OTHERS = OTHERS + tuple(
                    index_b[k] for k in other_keys_b if k not in other_keys_match
                )

            if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                # we hard-code a combination method where b can be a pair descriptor. this needs some work to be general and robust
                # note also that this assumes that structure, center are ordered in the same way in the centred and neighbor descriptors
                neighbor_slice = []
                smp_a, smp_b = 0, 0
                while smp_b < samples_b.shape[0]:
                    if samples_b[smp_b][["structure", "center"]] != samples_a[smp_a]:
                        smp_a += 1
                    neighbor_slice.append(smp_a)
                    smp_b += 1
                neighbor_slice = np.asarray(neighbor_slice)
            else:
                neighbor_slice = slice(None)

            # determines the properties that are in the select list
            sel_feats = []
            sel_idx = []
            sel_feats = (
                np.indices((len(properties_a), len(properties_b))).reshape(2, -1).T
            )

            prop_ids_a = []
            prop_ids_b = []
            for n_a, f_a in enumerate(properties_a):
                prop_ids_a.append(tuple(f_a) + (lam_a,))
            for n_b, f_b in enumerate(properties_b):
                prop_ids_b.append(tuple(f_b) + (lam_b,))
            prop_ids_a = np.asarray(prop_ids_a)
            prop_ids_b = np.asarray(prop_ids_b)
            sel_idx = np.hstack(
                [prop_ids_a[sel_feats[:, 0]], prop_ids_b[sel_feats[:, 1]]]
            )
            if len(sel_feats) == 0:
                continue
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                NU = order_a + order_b
                KEY = (
                    NU,
                    S,
                    L,
                ) + OTHERS
                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = block_b.samples
                    if grad_components is not None:
                        X_grads[KEY] = []
                        X_grad_samples[KEY] = block_b.gradient("positions").samples

                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )
                # do gradients, if they are present...
                if grad_components is not None:
                    grad_a = block_a.gradient("positions")
                    grad_b = block_b.gradient("positions")
                    grad_a_data = np.swapaxes(grad_a.data, 1, 2)
                    grad_b_data = np.swapaxes(grad_b.data, 1, 2)
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[grad_a.samples["sample"]][
                            neighbor_slice, :, sel_feats[:, 0]
                        ],
                        grad_b_data[..., sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[grad_b.samples["sample"]][:, :, sel_feats[:, 1]],
                        grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    )

                # now loop over the selected features to build the blocks

                X_idx[KEY].append(sel_idx)
                X_blocks[KEY].append(one_shot_blocks)
                if grad_components is not None:
                    X_grads[KEY].append(one_shot_grads)

    # turns data into sparse storage format (and dumps any empty block in the
    # process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[2]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
            ["spherical_harmonics_m"],
            np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1),
        )
        newblock = TensorBlock(
            # feature index must be last
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(
                feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)
            ),
        )
        if grad_components is not None:
            grad_data = np.swapaxes(np.concatenate(X_grads[KEY], axis=-1), 2, 1)
            newblock.add_gradient(
                "positions",
                data=grad_data,
                samples=X_grad_samples[KEY],
                components=[grad_components[0], sph_components],
            )
        nz_blk.append(newblock)
    X = TensorMap(
        Labels(
            ["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS,
            np.asarray(nz_idx, dtype=np.int32),
        ),
        nz_blk,
    )
    return X


def cg_increment(
    x_nu,
    x_1,
    clebsch_gordan=None,
    lcut=None,
    other_keys_match=None,
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""

    nu = x_nu.keys["order_nu"][0]

    feature_roots = _remove_suffix(x_1.block(0).properties.names)

    if nu == 1:
        feature_names = (
            [root + "_1" for root in feature_roots]
            + ["l_1"]
            + [root + "_2" for root in feature_roots]
            + ["l_2"]
        )
    else:
        feature_names = (
            [x_nu.block(0).properties.names]
            + ["k_" + str(nu + 1)]
            + [root + "_" + str(nu + 1) for root in feature_roots]
            + ["l_" + str(nu + 1)]
        )

    return cg_combine(
        x_nu,
        x_1,
        feature_names=feature_names,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
        other_keys_match=other_keys_match,
    )


def _remove_suffix(names, new_suffix=""):
    suffix = re.compile("_[0-9]?$")
    rname = []
    for name in names:
        match = suffix.search(name)
        if match is None:
            rname.append(name + new_suffix)
        else:
            rname.append(name[: match.start()] + new_suffix)
    return rname



def lambda_soap_vector(
    frames: list,
    rascal_hypers: dict,
    lambda_cut: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
    neighbor_species: Optional[Sequence[int]] = None,
    even_parity_only: bool = False,
) -> TensorMap:
    """
    Takes a list of frames of ASE loaded frames and a dict of Rascaline
    hyperparameters and generates a lambda-SOAP (i.e. nu=2) representation.

    Passing a subset of samples in `selected_samples` can be used to, for
    instance, only calculate the features for a subset of the strutcures passed
    in `frames`. For instance: `selected_samples = Labels(names=["structure"],
    values[4, 5, 6])` will only calculate the lambda-features for structures
    indexed by 4, 5, 6.

    :param frames: a list of structures generated by the ase.io function.
    :param rascal_hypers: a dict of hyperparameters used to calculate the atom
        density correlation calculated with rascaline SphericalExpansion
    :param lambda_cut: an int of the maximum lambda value to compute
        combinations for. If none, the 'max_angular' value in `rascal_hypers`
        will be used instead.
    :param selected_samples: a Labels object that defines which samples, as a
        subset of the total samples in `frames` (i.e. atomic environments or
        structures) to perform the calculation on.
    :param neighbor_species: a list of int that correspond to the atomic charges
        of all the neighbour species that you want to be in your properties (or
        features) dimension. This list may contain charges for atoms that don't
        appear in ``frames``, but are included anyway so that the one can
        enforce consistent properties dimension size with other lambda-feature
        vectors.
    :param even_parity_only: a bool that determines whether to only include the
        key/block pairs with even parity under rotation, i.e. sigma = +1.
        Defaults to false, where both parities are included.
    :param save_dir: a str of the absolute path to the directory where the
        TensorMap of the calculated lambda-SOAP representation and pickled
        ``rascal_hypers`` dict should be written. If none, the TensorMap will not be
        saved.

    :return: a TensorMap of the lambda-SOAP feature vector for the selected
        samples of the input frames.
    """
    # Generate Rascaline spherical expansion
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    if lambda_cut is None:
        lambda_cut = 2 * rascal_hypers["max_angular"]
    else:
        if lambda_cut > 2 * rascal_hypers["max_angular"]:
            raise ValueError(
                "As this function generates 2-body features (nu=2), "
                "`lambda_cut` must be <= 2 x rascal_hypers['max_angular'] "
                f"`rascal_hypers`. Received {lambda_cut}."
            )
    # Pre-calculate ClebschGordan coefficients
    cg = ClebschGordanReal(l_max=lambda_cut)

    # Generate descriptor via Spherical Expansion
    nu1 = calculator.compute(frames, selected_samples=selected_samples)

    # nu=1 features
    nu1 = acdc_standardize_keys(nu1)

    # Move "species_neighbor" sparse keys to properties with enforced atom
    # charges if ``neighbor_species`` is specified. This is required as the CG
    # iteration code currently does not handle neighbour species padding
    # automatically.
    keys_to_move = "species_neighbor"
    if neighbor_species is not None:
        keys_to_move = Labels(
            names=(keys_to_move,),
            values=np.array(neighbor_species).reshape(-1, 1),
        )
    nu1 = nu1.keys_to_properties(keys_to_move=keys_to_move)

    # Combined nu=1 features to generate nu=2 features. lambda-SOAP is defined
    # as just the nu=2 features.
    lsoap = cg_increment(
        nu1,
        nu1,
        clebsch_gordan=cg,
        lcut=lambda_cut,
        other_keys_match=["species_center"],
    )

    # Clean the lambda-SOAP TensorMap. Drop the order_nu key name as this is by
    # definition 2 for all keys.
    lsoap = equistore.remove_dimension(lsoap, axis="keys", name="order_nu")

    # Drop all odd parity keys/blocks
    if even_parity_only:
        keys_to_drop = Labels(
            names=lsoap.keys.names,
            values=lsoap.keys.values[lsoap.keys.column("inversion_sigma") == -1],
        )
        lsoap = equistore.drop_blocks(lsoap, keys=keys_to_drop)

        # Drop the inversion_sigma key name as this is now +1 for all blocks
        lsoap = equistore.remove_dimension(lsoap, axis="keys", name="inversion_sigma")

    return lsoap