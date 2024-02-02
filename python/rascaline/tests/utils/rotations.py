"""
Class for generating real Wigner-D matrices, and using them to rotate ASE frames
and TensorMaps of density coefficients in the spherical basis.
"""

from typing import Sequence

import pytest


ase = pytest.importorskip("ase")

import numpy as np  # noqa: E402
from metatensor import TensorBlock, TensorMap  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


try:
    import torch  # noqa: E402
    from torch import Tensor as TorchTensor  # noqa: E402
except ImportError:

    class TorchTensor:
        pass


# ===== Functions for transformations in the Cartesian basis =====


def cartesian_rotation(angles: Sequence[float]):
    """
    Returns a Cartesian rotation matrix in the appropriate convention (ZYZ,
    implicit rotations) to be consistent with the common Wigner D definition.

    `angles` correspond to the alpha, beta, gamma Euler angles in the ZYZ
    convention, in radians.
    """
    return Rotation.from_euler("ZYZ", angles).as_matrix()


def transform_frame_so3(frame: ase.Atoms, angles: Sequence[float]) -> ase.Atoms:
    """
    Transforms the positions and cell coordinates of an ASE frame by a SO(3)
    rigid rotation.
    """
    new_frame = frame.copy()

    # Build cartesian rotation matrix
    R = cartesian_rotation(angles)

    # Rotate its positions and cell
    new_frame.positions = new_frame.positions @ R.T
    new_frame.cell = new_frame.cell @ R.T

    return new_frame


def transform_frame_o3(frame: ase.Atoms, angles: Sequence[float]) -> ase.Atoms:
    """
    Transforms the positions and cell coordinates of an ASE frame by an O(3)
    rotation. This involves a rigid SO(3) rotation of the positions and cell
    according to the Euler `angles`, then an inversion by multiplying just the
    positions by -1.
    """
    new_frame = frame.copy()

    # Build cartesian rotation matrix
    R = cartesian_rotation(angles)

    # Rotate its positions and cell
    new_frame.positions = new_frame.positions @ R.T
    new_frame.cell = new_frame.cell @ R.T

    # Invert the atom positions
    new_frame.positions *= -1

    return new_frame


# ===== WignerDReal for transformations in the spherical basis =====


class WignerDReal:
    """
    A helper class to compute Wigner D matrices given the Euler angles of a rotation,
    and apply them to spherical harmonics (or coefficients). Built to function with
    real-valued coefficients.
    """

    def __init__(self, lmax: int, angles: Sequence[float] = None):
        """
        Initialize the WignerDReal class.

        :param lmax: int, the maximum angular momentum channel for which the
            Wigner D matrices are computed
        :param angles: Sequence[float], the alpha, beta, gamma Euler angles, in
            radians.
        """
        self.lmax = lmax
        # Randomly generate Euler angles between 0 and 2 pi if none are provided
        if angles is None:
            angles = np.random.uniform(size=(3)) * 2 * np.pi
        self.angles = angles
        self.rotation = cartesian_rotation(angles)

        r2c_mats = {}
        c2r_mats = {}
        for L in range(0, self.lmax + 1):
            r2c_mats[L] = np.hstack(
                [_r2c(np.eye(2 * L + 1)[i])[:, np.newaxis] for i in range(2 * L + 1)]
            )
            c2r_mats[L] = np.conjugate(r2c_mats[L]).T
        self.matrices = {}
        for L in range(0, self.lmax + 1):
            wig = _wigner_d(L, self.angles)
            self.matrices[L] = np.real(c2r_mats[L] @ np.conjugate(wig) @ r2c_mats[L])

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

        Required is the basis set definition specified by ``lmax`` and ``nmax``.
        This are dicts of the form:

            lmax = {symbol: lmax_value, ...}
            nmax = {(symbol, l): nmax_value, ...}

        where ``symbol`` is the chemical symbol of the atom, ``lmax_value`` is
        its corresponding max l channel value. For each combination of species
        symbol and lmax, there exists a max radial channel value ``nmax_value``.

        Then, the assumed ordering of basis function coefficients follows a
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
            for angular_l in range(sym_lmax + 1):
                # Get the number of radial functions for this species and l value
                sym_l_nmax = nmax[(symbol, angular_l)]
                # Get the Wigner D Matrix for this l value
                wig_mat = self.matrices[angular_l].T
                for _n in range(sym_l_nmax):
                    # Retrieve the irreducible spherical component
                    isc = coeffs[curr_idx : curr_idx + (2 * angular_l + 1)]
                    # Rotate the ISC and store
                    rot_isc = isc @ wig_mat
                    rot_vect[curr_idx : curr_idx + (2 * angular_l + 1)][:] = rot_isc[:]
                    # Update the start index for the next ISC
                    curr_idx += 2 * angular_l + 1

        return rot_vect

    def rotate_tensorblock(self, angular_l: int, block: TensorBlock) -> TensorBlock:
        """
        Rotates a TensorBlock ``block``, represented in the spherical basis,
        according to the Wigner D Real matrices for the given ``l`` value.
        Assumes the components of the block are [("spherical_harmonics_m",),].
        """
        # Get the Wigner matrix for this l value
        wig = self.matrices[angular_l].T

        # Copy the block
        block_rotated = block.copy()
        vals = block_rotated.values

        # Perform the rotation, either with numpy or torch, by taking the
        # tensordot product of the irreducible spherical components. Modify
        # in-place the values of the copied TensorBlock.
        if isinstance(vals, TorchTensor):
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

    def transform_tensormap_so3(self, tensor: TensorMap) -> TensorMap:
        """
        Transforms a TensorMap by a by an SO(3) rigid rotation using Wigner-D
        matrices.

        Assumes the input tensor follows the metadata structure consistent with
        those produce by rascaline.
        """
        # Retrieve the key and the position of the l value in the key names
        keys = tensor.keys
        idx_l_value = keys.names.index("spherical_harmonics_l")

        # Iterate over the blocks and rotate
        rotated_blocks = []
        for key in keys:
            # Retrieve the l value
            angular_l = key[idx_l_value]

            # Rotate the block and store
            rotated_blocks.append(self.rotate_tensorblock(angular_l, tensor[key]))

        return TensorMap(keys, rotated_blocks)

    def transform_tensormap_o3(self, tensor: TensorMap) -> TensorMap:
        """
        Transforms a TensorMap by a by an O(3) transformation: this involves an
        SO(3) rigid rotation using Wigner-D Matrices followed by an inversion.

        Assumes the input tensor follows the metadata structure consistent with
        those produce by rascaline.
        """
        # Retrieve the key and the position of the l value in the key names
        keys = tensor.keys
        idx_l_value = keys.names.index("spherical_harmonics_l")

        # Iterate over the blocks and rotate
        new_blocks = []
        for key in keys:
            # Retrieve the l value
            angular_l = key[idx_l_value]

            # Rotate the block
            new_block = self.rotate_tensorblock(angular_l, tensor[key])

            # Work out the inversion multiplier according to the convention
            inversion_multiplier = 1
            if key["spherical_harmonics_l"] % 2 == 1:
                inversion_multiplier *= -1

            # "inversion_sigma" may not be present if CG iterations haven't been
            # performed (i.e. nu=1 rascaline SphericalExpansion)
            if "inversion_sigma" in keys.names:
                if key["inversion_sigma"] == -1:
                    inversion_multiplier *= -1

            # Invert the block by applying the inversion multiplier
            new_block = TensorBlock(
                values=new_block.values * inversion_multiplier,
                samples=new_block.samples,
                components=new_block.components,
                properties=new_block.properties,
            )
            new_blocks.append(new_block)

        return TensorMap(keys, new_blocks)


# ===== Helper functions for WignerDReal


def _wigner_d(angular_l: int, angles: Sequence[float]) -> np.ndarray:
    """
    Computes the Wigner D matrix:
        D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.

    `angles` are the alpha, beta, gamma Euler angles (radians, ZYZ convention)
    and l the irrep.
    """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    return np.complex128(wigner_d(angular_l, *angles))


def _r2c(sp):
    """
    Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l
    """

    i_sqrt_2 = 1.0 / np.sqrt(2)

    angular_l = (len(sp) - 1) // 2  # infers l from the vector size
    rc = np.zeros(len(sp), dtype=np.complex128)
    rc[angular_l] = sp[angular_l]
    for m in range(1, angular_l + 1):
        rc[angular_l + m] = (
            (sp[angular_l + m] + 1j * sp[angular_l - m]) * i_sqrt_2 * (-1) ** m
        )
        rc[angular_l - m] = (sp[angular_l + m] - 1j * sp[angular_l - m]) * i_sqrt_2
    return rc
