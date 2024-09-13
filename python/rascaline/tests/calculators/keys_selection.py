import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from rascaline import RascalError
from rascaline.calculators import DummyCalculator, SphericalExpansion

from ..test_systems import SystemForTests


def _tensor_map_selection(label_type, keys, labels):
    blocks = []
    assert len(keys) == len(labels)
    for entries in labels:
        if label_type == "samples":
            blocks.append(
                TensorBlock(
                    values=np.empty((len(entries), 1)),
                    samples=entries,
                    components=[],
                    properties=Labels.single(),
                )
            )
        elif label_type == "properties":
            blocks.append(
                TensorBlock(
                    values=np.empty((1, len(entries))),
                    samples=Labels.single(),
                    components=[],
                    properties=entries,
                )
            )

    return TensorMap(keys, blocks)


def test_selection_existing():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    # no selection
    descriptor = calculator.compute(system, selected_keys=None)
    assert len(descriptor.keys) == 2
    assert descriptor.keys.values.tolist() == [[1], [8]]

    # Manually select the keys
    selected_keys = Labels(
        names=["center_type"],
        values=np.array([[1]], dtype=np.int32),
    )
    descriptor = calculator.compute(system, selected_keys=selected_keys)

    assert len(descriptor.keys) == 1
    assert descriptor.keys[0].values.tolist() == [1]


def test_selection_partial():
    system = SystemForTests()
    calculator = SphericalExpansion(
        cutoff=2.5,
        max_radial=1,
        max_angular=1,
        atomic_gaussian_width=0.2,
        radial_basis={"Gto": {}},
        cutoff_function={"ShiftedCosine": {"width": 0.5}},
        center_atom_weight=1.0,
    )

    # Manually select the keys
    selected_keys = Labels(
        names=["center_type"],
        values=np.array([[1]], dtype=np.int32),
    )
    descriptor = calculator.compute(system, selected_keys=selected_keys)

    assert len(descriptor.keys) == 4
    assert descriptor.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert descriptor.keys.values.tolist() == [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 8],
        [1, 1, 1, 8],
    ]


def test_select_key_not_in_systems():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    # Manually select the keys
    selected_keys = Labels(
        names=["center_type"],
        values=np.array([[4]], dtype=np.int32),
    )
    descriptor = calculator.compute(system, selected_keys=selected_keys)

    C_block = descriptor.block(center_type=4)
    assert C_block.values.shape == (0, 2)


def test_predefined_selection():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    selected_keys = Labels(
        names=["center_type"],
        values=np.array([[1]], dtype=np.int32),
    )

    keys = Labels(
        names=["center_type"],
        values=np.array([[1], [8]], dtype=np.int32),
    )

    # selection from TensorMap
    selected = [
        Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([[1, 0]], dtype=np.int32),
        ),
        Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([[0, 1]], dtype=np.int32),
        ),
    ]
    selected_properties = _tensor_map_selection("properties", keys, selected)

    descriptor = calculator.compute(
        system,
        selected_properties=selected_properties,
        selected_keys=selected_keys,
    )

    assert len(descriptor.keys) == 1
    H_block = descriptor.block(center_type=1)
    assert H_block.values.shape == (2, 1)
    assert np.all(H_block.values[0] == (2,))
    assert np.all(H_block.values[1] == (3,))


def test_name_errors():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    selected_keys = Labels(
        names=["bad_name"],
        values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
    )

    message = (
        "invalid parameter: 'bad_name' in keys selection is not "
        "part of the keys of this calculator"
    )
    with pytest.raises(RascalError, match=message):
        calculator.compute(system, selected_keys=selected_keys)


def test_key_errors():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    selected_keys = Labels(
        names=["center_type"],
        values=np.empty((0, 1), dtype=np.int32),
    )

    message = "invalid parameter: selected keys can not be empty"
    with pytest.raises(RascalError, match=message):
        calculator.compute(system, selected_keys=selected_keys)

    # in the case where both selected_properties/selected_samples and
    # selected_keys are given, the selected keys must be in the keys of the
    # predefined tensor_map
    selected_keys = Labels(
        names=["center_type"],
        values=np.array([[4]], dtype=np.int32),
    )

    keys = Labels(
        names=["center_type"],
        values=np.array([[1], [8]], dtype=np.int32),
    )

    selected = [
        Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([[1, 0]], dtype=np.int32),
        ),
        Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([[0, 1]], dtype=np.int32),
        ),
    ]
    selected_properties = _tensor_map_selection("properties", keys, selected)

    message = (
        "invalid parameter: expected a block for \\(center_type=4\\) in "
        "predefined properties selection"
    )
    with pytest.raises(RascalError, match=message):
        calculator.compute(
            system,
            selected_properties=selected_properties,
            selected_keys=selected_keys,
        )
