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

    def __init__(self, max_angular: int, angles: Sequence[float] = None):
        """
        Initialize the WignerDReal class.

        :param max_angular: int, the maximum angular momentum channel for which the
            Wigner D matrices are computed
        :param angles: Sequence[float], the alpha, beta, gamma Euler angles, in radians.
        """
        self.max_angular = max_angular
        # Randomly generate Euler angles between 0 and 2 pi if none are provided
        if angles is None:
            angles = np.random.uniform(size=(3)) * 2 * np.pi
        self.angles = angles
        self.rotation = cartesian_rotation(angles)

        r2c_mats = {}
        c2r_mats = {}
        for L in range(0, self.max_angular + 1):
            r2c_mats[L] = np.hstack(
                [_r2c(np.eye(2 * L + 1)[i])[:, np.newaxis] for i in range(2 * L + 1)]
            )
            c2r_mats[L] = np.conjugate(r2c_mats[L]).T
        self.matrices = {}
        for L in range(0, self.max_angular + 1):
            wig = _wigner_d(L, self.angles)
            self.matrices[L] = np.real(c2r_mats[L] @ np.conjugate(wig) @ r2c_mats[L])

    def rotate_tensorblock(self, o3_lambda: int, block: TensorBlock) -> TensorBlock:
        """
        Rotates a TensorBlock ``block``, represented in the spherical basis,
        according to the Wigner D Real matrices for the given ``l`` value.
        Assumes the components of the block are [("o3_mu",),].
        """
        # Get the Wigner matrix for this l value
        wig = self.matrices[o3_lambda].T

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
        those produce by featomic.
        """
        # Retrieve the key and the position of the l value in the key names
        keys = tensor.keys
        idx_l_value = keys.names.index("o3_lambda")

        # Iterate over the blocks and rotate
        rotated_blocks = []
        for key in keys:
            # Retrieve the l value
            o3_lambda = key[idx_l_value]

            # Rotate the block and store
            rotated_blocks.append(self.rotate_tensorblock(o3_lambda, tensor[key]))

        return TensorMap(keys, rotated_blocks)

    def transform_tensormap_o3(self, tensor: TensorMap) -> TensorMap:
        """
        Transforms a TensorMap by a by an O(3) transformation: this involves an
        SO(3) rigid rotation using Wigner-D Matrices followed by an inversion.

        Assumes the input tensor follows the metadata structure consistent with
        those produce by featomic.
        """
        # Retrieve the key and the position of the l value in the key names
        keys = tensor.keys
        idx_l_value = keys.names.index("o3_lambda")

        # Iterate over the blocks and rotate
        new_blocks = []
        for key in keys:
            # Retrieve the l value
            o3_lambda = key[idx_l_value]

            # Rotate the block
            new_block = self.rotate_tensorblock(o3_lambda, tensor[key])

            # Work out the inversion multiplier according to the convention
            inversion_multiplier = 1
            if key["o3_lambda"] % 2 == 1:
                inversion_multiplier *= -1

            # "o3_sigma" may not be present if CG iterations haven't been
            # performed (i.e. nu=1 featomic SphericalExpansion)
            if "o3_sigma" in keys.names:
                if key["o3_sigma"] == -1:
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


def _wigner_d(o3_lambda: int, angles: Sequence[float]) -> np.ndarray:
    """
    Computes the Wigner D matrix: ``D^l_{mm'}(alpha, beta, gamma)`` from sympy and
    converts it to numerical values.

    ``angles`` are the alpha, beta, gamma Euler angles (radians, ZYZ convention) and l
    the irrep.
    """
    try:
        from sympy.physics.wigner import wigner_d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Calculation of Wigner D matrices requires a sympy installation"
        )
    return np.complex128(wigner_d(o3_lambda, *angles))


def _r2c(sp):
    """
    Real to complex SPH. Assumes a block with 2l+1 reals corresponding
    to real SPH with m indices from -l to +l
    """

    i_sqrt_2 = 1.0 / np.sqrt(2)

    o3_lambda = (len(sp) - 1) // 2  # infers l from the vector size
    rc = np.zeros(len(sp), dtype=np.complex128)
    rc[o3_lambda] = sp[o3_lambda]
    for m in range(1, o3_lambda + 1):
        rc[o3_lambda + m] = (
            (sp[o3_lambda + m] + 1j * sp[o3_lambda - m]) * i_sqrt_2 * (-1) ** m
        )
        rc[o3_lambda - m] = (sp[o3_lambda + m] - 1j * sp[o3_lambda - m]) * i_sqrt_2
    return rc
