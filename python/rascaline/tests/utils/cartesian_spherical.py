import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from rascaline.utils.clebsch_gordan import cartesian_to_spherical


@pytest.fixture
def cartesian():
    # the first block is completely symmetric
    values_1 = np.random.rand(10, 4, 3, 3, 3, 2)
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
        values=np.random.rand(12, 6, 3, 3, 3, 7),
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


def test_cartesian_to_spherical_rank_1(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_2", "xyz_3"]
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "key"]

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1"],
        keep_l_in_keys=True,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_1", "key"]


def test_cartesian_to_spherical_rank_2(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_2"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_3"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "key"],
        np.array(
            [
                # only o3_sigma=1 in the symmetric block
                [0, 1, 0],
                [2, 1, 0],
                # all o3_sigma in the non-symmetric block
                [0, 1, 1],
                [1, -1, 1],
                [2, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2"],
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_2", "l_1", "key"]
    # all blocks are kept
    assert len(spherical.keys) == 6


def test_cartesian_to_spherical_rank_3(cartesian):
    spherical = cartesian_to_spherical(
        cartesian, components=["xyz_1", "xyz_2", "xyz_3"]
    )

    assert spherical.component_names == ["other", "o3_mu"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "l_3", "k_1", "l_2", "l_1", "key"],
        np.array(
            [
                # only o3_sigma=1 for the symmetric block, but there are multiple "path"
                # ("l_3", "k_1", "l_2", "l_1") that lead to o3_lambda=1
                [1, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 2, 1, 1, 0],
                [3, 1, 1, 2, 1, 1, 0],
                # all possible o3_sigma for the non-symmetric block
                [1, 1, 1, 0, 1, 1, 1],
                [0, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [2, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [2, -1, 1, 2, 1, 1, 1],
                [3, 1, 1, 2, 1, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2", "xyz_3"],
        remove_blocks_threshold=None,
    )
    # all blocks are kept, even those with norm=0
    assert len(spherical.keys) == 14


@pytest.mark.parametrize("cg_backend", ["python-dense", "python-sparse"])
@pytest.mark.parametrize(
    "components", [["xyz_1"], ["xyz_1", "xyz_12"], ["xyz_1", "xyz_2", "xyz_3"]]
)
def test_cartesian_to_spherical_and_back(cartesian, components, cg_backend):
    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2", "xyz_3"],
        keep_l_in_keys=True,
        cg_backend=cg_backend,
    )

    assert "o3_lambda" in spherical.keys.names
    # TODO: check for identity after spherical_to_cartesian


def test_cartesian_to_spherical_errors(cartesian):
    message = "`components` should be a list, got <class 'str'>"
    with pytest.raises(TypeError, match=message):
        cartesian_to_spherical(cartesian, components="xyz_1")

    message = "'1' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=[1, 2])

    message = "'not_there' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["not_there"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_2', 'xyz_1'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_2", "xyz_1"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_1', 'xyz_3'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_3"])

    message = (
        "component 'other' in block for \\(key=0\\) should have \\[0, 1, 2\\] "
        "as values, got \\[0, 1, 2, 3\\] instead"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["other"])

    message = "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(
            cartesian,
            components=["xyz_1", "xyz_2", "xyz_3"],
            keep_l_in_keys=False,
        )
