# -*- coding: utf-8 -*-
from rascaline import Descriptor
import rascaline
import unittest


class TestEmptyDescriptor(unittest.TestCase):
    def test_values(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.values.shape, (0, 0))

    def test_gradients(self):
        descriptor = Descriptor()
        self.assertEqual(descriptor.gradients.shape, (0, 0))

    def test_environments(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.environments), 0)

    def test_features(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.features), 0)

    def test_grad_environments(self):
        descriptor = Descriptor()
        self.assertEqual(len(descriptor.gradients_environments), 0)
