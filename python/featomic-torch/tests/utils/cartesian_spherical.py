import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from featomic.torch.utils.clebsch_gordan import cartesian_to_spherical


@pytest.fixture
def cartesian():
    # the first block is completely symmetric
    values_1 = torch.rand(10, 4, 3, 3, 3, 2, dtype=torch.float64)
    values_1[:, :, 0, 1, 0, :] = values_1[:, :, 0, 0, 1, :]
    values_1[:, :, 1, 0, 0, :] = values_1[:, :, 0, 0, 1, :]

    values_1[:, :, 0, 2, 0, :] = values_1[:, :, 0, 0, 2, :]
    values_1[:, :, 2, 0, 0, :] = values_1[:, :, 0, 0, 2, :]

    values_1[:, :, 1, 0, 1, :] = values_1[:, :, 0, 1, 1, :]
    values_1[:, :, 1, 1, 0, :] = values_1[:, :, 0, 1, 1, :]

    values_1[:, :, 2, 0, 2, :] = values_1[:, :, 0, 2, 2, :]
    values_1[:, :, 2, 2, 0, :] = values_1[:, :, 0, 2, 2, :]

    values_1[:, :, 2, 1, 2, :] = values_1[:, :, 2, 2, 1, :]
    values_1[:, :, 1, 2, 2, :] = values_1[:, :, 2, 2, 1, :]

    values_1[:, :, 1, 2, 1, :] = values_1[:, :, 1, 1, 2, :]
    values_1[:, :, 2, 1, 1, :] = values_1[:, :, 1, 1, 2, :]

    values_1[:, :, 0, 2, 1, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 2, 0, 1, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 1, 0, 2, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 1, 2, 0, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 2, 1, 0, :] = values_1[:, :, 0, 1, 2, :]

    block_1 = TensorBlock(
        values=values_1,
        samples=Labels.range("s", 10),
        components=[
            Labels.range("other", 4),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 2),
    )

    # second block does not have any specific symmetry
    block_2 = TensorBlock(
        values=torch.rand(12, 6, 3, 3, 3, 7, dtype=torch.float64),
        samples=Labels.range("s", 12),
        components=[
            Labels.range("other", 6),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 7),
    )

    return TensorMap(Labels.range("key", 2), [block_1, block_2])


def test_torch_script():
    torch.jit.script(cartesian_to_spherical)


def test_cartesian_to_spherical(cartesian):
    # rank 1
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_2", "xyz_3"]
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "key"]
    assert len(spherical.keys) == 2

    # rank 2
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_2"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_3"]
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "key"]
    assert len(spherical.keys) == 5

    # rank 3
    spherical = cartesian_to_spherical(
        cartesian, components=["xyz_1", "xyz_2", "xyz_3"]
    )

    assert spherical.component_names == ["other", "o3_mu"]
    assert spherical.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "l_3",
        "k_1",
        "l_2",
        "l_1",
        "key",
    ]
    assert len(spherical.keys) == 10
