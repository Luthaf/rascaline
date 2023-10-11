import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from rascaline import RascalError
from rascaline.calculators import DummyCalculator

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


def test_selection():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    # Manually constructing the selected samples
    selected_samples = Labels(
        names=["structure", "center"],
        values=np.array([(0, 0), (0, 3), (0, 1)], dtype=np.int32),
    )
    descriptor = calculator.compute(
        system, use_native_system=False, selected_samples=selected_samples
    )

    H_block = descriptor.block(species_center=1)
    assert H_block.values.shape == (2, 2)
    assert np.all(H_block.values[0] == (2, 11))
    assert np.all(H_block.values[1] == (3, 13))

    O_block = descriptor.block(species_center=8)
    assert O_block.values.shape == (1, 2)
    assert np.all(O_block.values[0] == (5, 5))


def test_subset_variables():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    # Only a subset of the variables defined
    selected_samples = Labels(
        names=["center"],
        values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
    )
    descriptor = calculator.compute(
        system, use_native_system=False, selected_samples=selected_samples
    )

    H_block = descriptor.block(species_center=1)
    assert H_block.values.shape == (2, 2)
    assert np.all(H_block.values[0] == (2, 11))
    assert np.all(H_block.values[1] == (3, 13))

    O_block = descriptor.block(species_center=8)
    assert O_block.values.shape == (1, 2)
    assert np.all(O_block.values[0] == (5, 5))


def test_empty_selection():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    # empty selected samples
    selected_samples = Labels(
        names=["center"],
        values=np.empty((0, 1), dtype=np.int32),
    )
    descriptor = calculator.compute(
        system, use_native_system=False, selected_samples=selected_samples
    )

    H_block = descriptor.block(species_center=1)
    assert H_block.values.shape == (0, 2)

    O_block = descriptor.block(species_center=8)
    assert O_block.values.shape == (0, 2)


def test_predefined_selection():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    keys = Labels(
        names=["species_center"],
        values=np.array([[1], [8]], dtype=np.int32),
    )

    # selection from TensorMap
    selected = [
        Labels(
            names=["structure", "center"],
            values=np.array([[0, 1]], dtype=np.int32),
        ),
        Labels(
            names=["structure", "center"],
            values=np.array([[0, 3]], dtype=np.int32),
        ),
    ]
    selected_samples = _tensor_map_selection("samples", keys, selected)

    descriptor = calculator.compute(
        system, use_native_system=False, selected_samples=selected_samples
    )

    H_block = descriptor.block(species_center=1)
    assert H_block.values.shape == (1, 2)
    assert np.all(H_block.values[0] == (3, 13))

    O_block = descriptor.block(species_center=8)
    assert O_block.values.shape == (1, 2)
    assert np.all(O_block.values[0] == (5, 5))


def test_errors():
    system = SystemForTests()
    calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

    samples = Labels(
        names=["bad_name"],
        values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
    )

    message = (
        "invalid parameter: 'bad_name' in samples selection is not one "
        "of the samples of this calculator"
    )
    with pytest.raises(RascalError, match=message):
        calculator.compute(system, use_native_system=False, selected_samples=samples)
