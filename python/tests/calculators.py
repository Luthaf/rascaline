import unittest

import numpy as np
from equistore import Labels, TensorBlock, TensorMap

from rascaline import RascalError, SortedDistances
from rascaline.calculators import DummyCalculator

from test_systems import TestSystem


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


class TestDummyCalculator(unittest.TestCase):
    def test_name(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo")
        self.assertEqual(
            calculator.name,
            "dummy test calculator with cutoff: 3.2 - delta: 12 - name: foo",
        )

        self.assertEqual(calculator.c_name, "dummy_calculator")

        # very long name, checking that we can pass large string back and forth
        name = "abc" * 2048
        calculator = DummyCalculator(cutoff=3.2, delta=12, name=name)
        self.assertEqual(
            calculator.name,
            f"dummy test calculator with cutoff: 3.2 - delta: 12 - name: {name}",
        )

    def test_parameters(self):
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="foo")
        self.assertEqual(
            calculator.parameters,
            """{"cutoff": 3.2, "delta": 12, "name": "foo"}""",
        )

    def test_bad_parameters(self):
        message = (
            'json error: invalid type: string "12", expected isize at line 1 column 29'
        )

        with self.assertRaisesRegex(Exception, message):
            _ = DummyCalculator(cutoff=3.2, delta="12", name="foo")

    def test_compute(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
        descriptor = calculator.compute(
            system, use_native_system=False, gradients=["positions"]
        )

        self.assertEqual(len(descriptor.keys), 2)
        self.assertEqual(descriptor.keys.names, ("species_center",))
        self.assertEqual(tuple(descriptor.keys[0]), (1,))
        self.assertEqual(tuple(descriptor.keys[1]), (8,))

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 2))
        self.assertTrue(np.all(H_block.values[0] == (2, 1)))
        self.assertTrue(np.all(H_block.values[1] == (3, 3)))

        self.assertEqual(len(H_block.samples), 2)
        self.assertEqual(H_block.samples.names, ("structure", "center"))
        self.assertEqual(tuple(H_block.samples[0]), (0, 0))
        self.assertEqual(tuple(H_block.samples[1]), (0, 1))

        self.assertEqual(len(H_block.components), 0)

        self.assertEqual(len(H_block.properties), 2)
        self.assertEqual(H_block.properties.names, ("index_delta", "x_y_z"))
        self.assertEqual(tuple(H_block.properties[0]), (1, 0))
        self.assertEqual(tuple(H_block.properties[1]), (0, 1))

        gradient = H_block.gradient("positions")
        self.assertEqual(gradient.data.shape, (5, 3, 2))
        for i in range(gradient.data.shape[0]):
            self.assertTrue(np.all(gradient.data[i, 0, :] == (0, 1)))
            self.assertTrue(np.all(gradient.data[i, 1, :] == (0, 1)))
            self.assertTrue(np.all(gradient.data[i, 2, :] == (0, 1)))

        self.assertEqual(len(gradient.samples), 5)
        self.assertEqual(gradient.samples.names, ("sample", "structure", "atom"))
        self.assertEqual(tuple(gradient.samples[0]), (0, 0, 0))
        self.assertEqual(tuple(gradient.samples[1]), (0, 0, 1))
        self.assertEqual(tuple(gradient.samples[2]), (1, 0, 0))
        self.assertEqual(tuple(gradient.samples[3]), (1, 0, 1))
        self.assertEqual(tuple(gradient.samples[4]), (1, 0, 2))

        self.assertEqual(len(gradient.components), 1)
        component = gradient.components[0]
        self.assertEqual(len(component), 3)
        self.assertEqual(component.names, ("direction",))
        self.assertEqual(tuple(component[0]), (0,))
        self.assertEqual(tuple(component[1]), (1,))
        self.assertEqual(tuple(component[2]), (2,))

        self.assertEqual(len(gradient.properties), 2)
        self.assertEqual(gradient.properties.names, ("index_delta", "x_y_z"))
        self.assertEqual(tuple(gradient.properties[0]), (1, 0))
        self.assertEqual(tuple(gradient.properties[1]), (0, 1))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (2, 2))
        self.assertTrue(np.all(O_block.values[0] == (4, 6)))
        self.assertTrue(np.all(O_block.values[1] == (5, 5)))

    def test_compute_multiple_systems(self):
        systems = [TestSystem(), TestSystem(), TestSystem()]
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
        descriptor = calculator.compute(systems, use_native_system=False)

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (6, 2))
        expected = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertTrue(
            np.all(H_block.samples.view(np.int32).reshape(6, 2) == expected)
        )

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (6, 2))


class TestComputePartialSamples(unittest.TestCase):
    def test_selection(self):
        system = TestSystem()
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
        self.assertEqual(H_block.values.shape, (2, 2))
        self.assertTrue(np.all(H_block.values[0] == (2, 1)))
        self.assertTrue(np.all(H_block.values[1] == (3, 3)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (1, 2))
        self.assertTrue(np.all(O_block.values[0] == (5, 5)))

    def test_subset_variables(self):
        system = TestSystem()
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
        self.assertEqual(H_block.values.shape, (2, 2))
        self.assertTrue(np.all(H_block.values[0] == (2, 1)))
        self.assertTrue(np.all(H_block.values[1] == (3, 3)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (1, 2))
        self.assertTrue(np.all(O_block.values[0] == (5, 5)))

    def test_empty_selection(self):
        system = TestSystem()
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
        self.assertEqual(H_block.values.shape, (0, 2))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (0, 2))

    def test_predefined_selection(self):
        system = TestSystem()
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
        self.assertEqual(H_block.values.shape, (1, 2))
        self.assertTrue(np.all(H_block.values[0] == (3, 3)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (1, 2))
        self.assertTrue(np.all(O_block.values[0] == (5, 5)))

    def test_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        samples = Labels(
            names=["bad_name"],
            values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_samples=samples
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: 'bad_name' in samples selection is not one "
            "of the samples of this calculator",
        )


class TestComputePartialProperties(unittest.TestCase):
    def test_selection(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        # Manually constructing the selected properties
        selected_properties = Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([[1, 0]], dtype=np.int32),
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_properties=selected_properties
        )

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 1))
        self.assertTrue(np.all(H_block.values[0] == (2,)))
        self.assertTrue(np.all(H_block.values[1] == (3,)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (2, 1))
        self.assertTrue(np.all(O_block.values[0] == (4,)))
        self.assertTrue(np.all(O_block.values[1] == (5,)))

    def test_subset_variables(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        # Only a subset of the variables defined
        selected_properties = Labels(
            names=["index_delta"],
            values=np.array([[1]], dtype=np.int32),
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_properties=selected_properties
        )

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 1))
        self.assertTrue(np.all(H_block.values[0] == (2,)))
        self.assertTrue(np.all(H_block.values[1] == (3,)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (2, 1))
        self.assertTrue(np.all(O_block.values[0] == (4,)))
        self.assertTrue(np.all(O_block.values[1] == (5,)))

    def test_empty_selection(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        # empty selected features
        selected_properties = Labels(
            names=["index_delta", "x_y_z"],
            values=np.array([], dtype=np.int32).reshape(0, 2),
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_properties=selected_properties
        )

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 0))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (2, 0))

    def test_predefined_selection(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        keys = Labels(
            names=["species_center"],
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
            system, use_native_system=False, selected_properties=selected_properties
        )

        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 1))
        self.assertTrue(np.all(H_block.values[0] == (2,)))
        self.assertTrue(np.all(H_block.values[1] == (3,)))

        O_block = descriptor.block(species_center=8)
        self.assertEqual(O_block.values.shape, (2, 1))
        self.assertTrue(np.all(O_block.values[0] == (6,)))
        self.assertTrue(np.all(O_block.values[1] == (5,)))

    def test_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        selected_properties = Labels(
            names=["bad_name"],
            values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system,
                use_native_system=False,
                selected_properties=selected_properties,
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: 'bad_name' in properties selection is not "
            "one of the properties of this calculator",
        )


class TestComputeSelectedKeys(unittest.TestCase):
    def test_selection_existing(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        # Manually select the keys
        selected_keys = Labels(
            names=["species_center"],
            values=np.array([[1]], dtype=np.int32),
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_keys=selected_keys
        )

        self.assertEqual(len(descriptor.keys), 1)
        self.assertEqual(tuple(descriptor.keys[0]), (1,))

    def test_select_key_not_in_systems(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        # Manually select the keys
        selected_keys = Labels(
            names=["species_center"],
            values=np.array([[4]], dtype=np.int32),
        )
        descriptor = calculator.compute(
            system, use_native_system=False, selected_keys=selected_keys
        )

        C_block = descriptor.block(species_center=4)
        self.assertEqual(C_block.values.shape, (0, 2))

    def test_predefined_selection(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        selected_keys = Labels(
            names=["species_center"],
            values=np.array([[1]], dtype=np.int32),
        )

        keys = Labels(
            names=["species_center"],
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
            use_native_system=False,
            selected_properties=selected_properties,
            selected_keys=selected_keys,
        )

        self.assertEqual(len(descriptor.keys), 1)
        H_block = descriptor.block(species_center=1)
        self.assertEqual(H_block.values.shape, (2, 1))
        self.assertTrue(np.all(H_block.values[0] == (2,)))
        self.assertTrue(np.all(H_block.values[1] == (3,)))

    def test_name_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        selected_keys = Labels(
            names=["bad_name"],
            values=np.array([0, 3, 1], dtype=np.int32).reshape(3, 1),
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_keys=selected_keys
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: names for the keys of the calculator "
            "[species_center] and selected keys [bad_name] do not match",
        )

    def test_key_errors(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")

        selected_keys = Labels(
            names=["species_center"],
            values=np.empty((0, 1), dtype=np.int32),
        )

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system, use_native_system=False, selected_keys=selected_keys
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: selected keys can not be empty",
        )

        # in the case where both selected_properties/selected_samples and
        # selected_keys are given, the selected keys must be in the keys of the
        # predefined tensor_map
        selected_keys = Labels(
            names=["species_center"],
            values=np.array([[4]], dtype=np.int32),
        )

        keys = Labels(
            names=["species_center"],
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

        with self.assertRaises(RascalError) as cm:
            calculator.compute(
                system,
                use_native_system=False,
                selected_properties=selected_properties,
                selected_keys=selected_keys,
            )

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: expected a key [4] in predefined properties selection",
        )


class TestSortedDistances(unittest.TestCase):
    def test_name(self):
        calculator = SortedDistances(
            cutoff=3.5, max_neighbors=12, separate_neighbor_species=False
        )
        self.assertEqual(calculator.name, "sorted distances vector")
        self.assertEqual(calculator.c_name, "sorted_distances")

    def test_parameters(self):
        calculator = SortedDistances(
            cutoff=3.5, max_neighbors=12, separate_neighbor_species=False
        )
        self.assertEqual(
            calculator.parameters,
            """{"cutoff": 3.5, "max_neighbors": 12, "separate_neighbor_species": false}""",  # noqa
        )


if __name__ == "__main__":
    unittest.main()
