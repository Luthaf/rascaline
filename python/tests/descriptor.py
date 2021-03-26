# -*- coding: utf-8 -*-
import unittest
import numpy as np

from rascaline import Descriptor
from rascaline.calculator import DummyCalculator

from test_systems import TestSystem, EmptySystem


class TestEmptyDescriptor(unittest.TestCase):
    def test_values(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.values.shape, (0, 0))

    def test_gradients(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.gradients, None)

    def test_environments(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.environments), 0)

    def test_features(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.features), 0)

    def test_gradients_environments(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.gradients_environments), 0)


class TestDummyDescriptor(unittest.TestCase):
    def test_values(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system)

        values = descriptor.values
        self.assertEqual(values.shape, (4, 2))
        self.assertTrue(np.all(values[0] == (12, 1)))
        self.assertTrue(np.all(values[1] == (13, 3)))
        self.assertTrue(np.all(values[2] == (14, 6)))
        self.assertTrue(np.all(values[3] == (15, 5)))

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            values[0] = (3, 4)

        self.assertEqual(descriptor.gradients, None)

    def test_gradients(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system)
        self.assertEqual(descriptor.gradients, None)

        system = EmptySystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
        descriptor = calculator.compute(system)
        self.assertEqual(descriptor.gradients.shape, (0, 2))

        system = TestSystem()
        descriptor = calculator.compute(system)
        gradients = descriptor.gradients
        self.assertEqual(gradients.shape, (18, 2))
        for i in range(18):
            self.assertTrue(np.all(gradients[i] == (0, 1)))

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            gradients[0] = (3, 4)

    def test_environments(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system)

        environments = descriptor.environments
        self.assertEqual(len(environments), 4)

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            environments[0] = (3, 4)

        self.assertTrue(np.all(environments["structure"] == [0, 0, 0, 0]))
        self.assertTrue(np.all(environments["center"] == [0, 1, 2, 3]))

        # view & reshape for easier direct comparison of values
        # numpy only consider structured arrays to be equal if they have
        # the same dtype
        environments = environments.view(dtype=np.int32).reshape(
            (environments.shape[0], -1)
        )
        self.assertTrue(np.all(environments[0] == [0, 0]))
        self.assertTrue(np.all(environments[1] == [0, 1]))
        self.assertTrue(np.all(environments[2] == [0, 2]))
        self.assertTrue(np.all(environments[3] == [0, 3]))

    def test_gradient_indexes(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system)
        self.assertEqual(len(descriptor.gradients_environments), 0)

        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
        descriptor = calculator.compute(system)
        gradients_environments = descriptor.gradients_environments
        self.assertEqual(len(gradients_environments), 18)

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            gradients_environments[0] = (3, 4)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(np.all(gradients_environments["structure"] == expected))

        expected = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
        self.assertTrue(np.all(gradients_environments["center"] == expected))

        expected = [1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, 2, 2, 2]
        self.assertTrue(np.all(gradients_environments["neighbor"] == expected))

        expected = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.assertTrue(np.all(gradients_environments["spatial"] == expected))

        # view & reshape for easier direct comparison of values
        # numpy only consider structured arrays to be equal if they have
        # the same dtype
        gradients_environments = gradients_environments.view(dtype=np.int32).reshape(
            (gradients_environments.shape[0], -1)
        )

        self.assertTrue(np.all(gradients_environments[0] == [0, 0, 1, 0]))
        self.assertTrue(np.all(gradients_environments[1] == [0, 0, 1, 1]))
        self.assertTrue(np.all(gradients_environments[2] == [0, 0, 1, 2]))
        self.assertTrue(np.all(gradients_environments[3] == [0, 1, 0, 0]))
        self.assertTrue(np.all(gradients_environments[4] == [0, 1, 0, 1]))
        self.assertTrue(np.all(gradients_environments[5] == [0, 1, 0, 2]))
        self.assertTrue(np.all(gradients_environments[6] == [0, 1, 2, 0]))
        self.assertTrue(np.all(gradients_environments[7] == [0, 1, 2, 1]))
        self.assertTrue(np.all(gradients_environments[8] == [0, 1, 2, 2]))
        self.assertTrue(np.all(gradients_environments[9] == [0, 2, 1, 0]))
        self.assertTrue(np.all(gradients_environments[10] == [0, 2, 1, 1]))
        self.assertTrue(np.all(gradients_environments[11] == [0, 2, 1, 2]))
        self.assertTrue(np.all(gradients_environments[12] == [0, 2, 3, 0]))
        self.assertTrue(np.all(gradients_environments[13] == [0, 2, 3, 1]))
        self.assertTrue(np.all(gradients_environments[14] == [0, 2, 3, 2]))
        self.assertTrue(np.all(gradients_environments[15] == [0, 3, 2, 0]))
        self.assertTrue(np.all(gradients_environments[16] == [0, 3, 2, 1]))
        self.assertTrue(np.all(gradients_environments[17] == [0, 3, 2, 2]))

    def test_features(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=False)
        descriptor = calculator.compute(system)

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
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=12, name="", gradients=True)
        descriptor = calculator.compute(system)

        self.assertEqual(descriptor.values.shape, (4, 2))
        self.assertEqual(descriptor.gradients.shape, (18, 2))

        descriptor.densify("center")

        self.assertEqual(descriptor.values.shape, (1, 8))
        self.assertEqual(descriptor.gradients.shape, (12, 8))
