# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline import Descriptor, RascalError
from rascaline.calculators import DummyCalculator

from test_systems import TestSystem, EmptySystem


class TestEmptyDescriptor(unittest.TestCase):
    def test_values(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.values.shape, (0, 0))

    def test_gradients(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.gradients, None)

    def test_samples(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.samples), 0)

    def test_features(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.features), 0)

    def test_gradients_samples(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.gradients_samples), 0)


class TestDummyDescriptor(unittest.TestCase):
    def test_values(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system, use_native_system=False)

        values = descriptor.values
        self.assertEqual(values.shape, (4, 2))
        self.assertTrue(np.all(values[0] == (12, 1)))
        self.assertTrue(np.all(values[1] == (13, 3)))
        self.assertTrue(np.all(values[2] == (14, 6)))
        self.assertTrue(np.all(values[3] == (15, 5)))

        # we can write to the array
        values[0] = (3, 4)
        self.assertTrue(np.all(values[0] == (3, 4)))
        self.assertTrue(np.all(values[1] == (13, 3)))

        self.assertEqual(descriptor.gradients, None)

    def test_gradients(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system, use_native_system=False)
        self.assertEqual(descriptor.gradients, None)

        system = EmptySystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
        descriptor = calculator.compute(system, use_native_system=False)
        self.assertEqual(descriptor.gradients.shape, (0, 2))

        system = TestSystem()
        descriptor = calculator.compute(system, use_native_system=False)
        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (18, 2))
        for i in range(18):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

        # we can write to the array
        gradients[0] = (3, 4)
        self.assertTrue(np.all(gradients[0] == (3, 4)))
        self.assertTrue(np.all(gradients[1] == (0, 1)))

    def test_samples(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system, use_native_system=False)

        samples = descriptor.samples
        self.assertEqual(len(samples), 4)

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            samples[0] = (3, 4)

        self.assertTrue(np.all(samples["structure"] == [0, 0, 0, 0]))
        self.assertTrue(np.all(samples["center"] == [0, 1, 2, 3]))

        # view & reshape for easier direct comparison of values
        # numpy only consider structured arrays to be equal if they have
        # the same dtype
        samples = samples.view(dtype=np.int32).reshape((samples.shape[0], -1))
        self.assertTrue(np.all(samples[0] == [0, 0]))
        self.assertTrue(np.all(samples[1] == [0, 1]))
        self.assertTrue(np.all(samples[2] == [0, 2]))
        self.assertTrue(np.all(samples[3] == [0, 3]))

    def test_gradient_indexes(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system, use_native_system=False)
        self.assertEqual(len(descriptor.gradients_samples), 0)

        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
        descriptor = calculator.compute(system, use_native_system=False)
        gradients_samples = descriptor.gradients_samples
        self.assertEqual(len(gradients_samples), 18)

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            gradients_samples[0] = (3, 4)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(np.all(gradients_samples["structure"] == expected))

        expected = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
        self.assertTrue(np.all(gradients_samples["center"] == expected))

        expected = [1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, 2, 2, 2]
        self.assertTrue(np.all(gradients_samples["neighbor"] == expected))

        expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.assertTrue(np.all(gradients_samples["spatial"] == expected))

        # view & reshape for easier direct comparison of values
        # numpy only consider structured arrays to be equal if they have
        # the same dtype
        gradients_samples = gradients_samples.view(dtype=np.int32).reshape(
            (gradients_samples.shape[0], -1)
        )

        self.assertTrue(np.all(gradients_samples[0] == [0, 0, 1, 0]))
        self.assertTrue(np.all(gradients_samples[1] == [0, 0, 1, 1]))
        self.assertTrue(np.all(gradients_samples[2] == [0, 0, 1, 2]))
        self.assertTrue(np.all(gradients_samples[3] == [0, 1, 0, 0]))
        self.assertTrue(np.all(gradients_samples[4] == [0, 1, 0, 1]))
        self.assertTrue(np.all(gradients_samples[5] == [0, 1, 0, 2]))
        self.assertTrue(np.all(gradients_samples[6] == [0, 1, 2, 0]))
        self.assertTrue(np.all(gradients_samples[7] == [0, 1, 2, 1]))
        self.assertTrue(np.all(gradients_samples[8] == [0, 1, 2, 2]))
        self.assertTrue(np.all(gradients_samples[9] == [0, 2, 1, 0]))
        self.assertTrue(np.all(gradients_samples[10] == [0, 2, 1, 1]))
        self.assertTrue(np.all(gradients_samples[11] == [0, 2, 1, 2]))
        self.assertTrue(np.all(gradients_samples[12] == [0, 2, 3, 0]))
        self.assertTrue(np.all(gradients_samples[13] == [0, 2, 3, 1]))
        self.assertTrue(np.all(gradients_samples[14] == [0, 2, 3, 2]))
        self.assertTrue(np.all(gradients_samples[15] == [0, 3, 2, 0]))
        self.assertTrue(np.all(gradients_samples[16] == [0, 3, 2, 1]))
        self.assertTrue(np.all(gradients_samples[17] == [0, 3, 2, 2]))

    def test_features(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system, use_native_system=False)

        features = descriptor.features
        self.assertEqual(len(features), 2)

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            features[0] = (3, 4)

        self.assertTrue(np.all(features["index_delta"] == [1, 0]))
        self.assertTrue(np.all(features["x_y_z"] == [0, 1]))

        # view & reshape for easier direct comparison of values
        # numpy only consider structured arrays to be equal if they have
        # the same dtype
        features = features.view(dtype=np.int32).reshape((features.shape[0], -1))
        self.assertTrue(np.all(features[0] == [1, 0]))
        self.assertTrue(np.all(features[1] == [0, 1]))

    def test_densify(self):
        def compute_descriptor():
            system = TestSystem()
            calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
            return calculator.compute(system, use_native_system=False)

        descriptor = compute_descriptor()

        self.assertEqual(descriptor.values.shape, (4, 2))
        self.assertEqual(descriptor.gradients.shape, (18, 2))

        descriptor.densify("center")
        self.assertEqual(descriptor.values.shape, (1, 8))
        self.assertEqual(descriptor.gradients.shape, (12, 8))

        descriptor = compute_descriptor()
        descriptor.densify("center", [1, 3, 8])
        self.assertEqual(descriptor.values.shape, (1, 6))
        self.assertEqual(descriptor.gradients.shape, (6, 6))

        descriptor = compute_descriptor()
        with self.assertRaises(RascalError) as cm:
            descriptor.densify("not there")

        self.assertEqual(
            str(cm.exception),
            "invalid parameter: can not densify along 'not there' which is not "
            + "present in the samples: [structure, center]",
        )

        descriptor = compute_descriptor()
        densified_positions = descriptor.densify_values("center")
        self.assertEqual(descriptor.values.shape, (1, 8))
        # gradients were not changed
        self.assertEqual(descriptor.gradients.shape, (18, 2))

        self.assertEqual(densified_positions.shape, (18,))
        self.assertEqual(densified_positions[0]["old_sample"], 0)
        self.assertEqual(densified_positions[0]["new_sample"], 0)
        self.assertEqual(densified_positions[0]["feature_block"], 0)

        self.assertEqual(densified_positions[6]["old_sample"], 6)
        self.assertEqual(densified_positions[6]["new_sample"], 6)
        self.assertEqual(densified_positions[6]["feature_block"], 1)

        self.assertEqual(densified_positions[15]["old_sample"], 15)
        self.assertEqual(densified_positions[15]["new_sample"], 6)
        self.assertEqual(densified_positions[15]["feature_block"], 3)
