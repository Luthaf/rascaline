# -*- coding: utf-8 -*-
import unittest

import numpy as np
from equistore import TensorBlock, TensorMap
from skcosmo.feature_selection import FPS as FPS_f

from rascaline.calculators import DummyCalculator
from rascaline.transformer import Transformer

from test_systems import TestSystem


class TestTransformer(unittest.TestCase):
    def test_keys_to_samples(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
        descriptor = calculator.compute(system)
        tr = Transformer(
            selector=FPS_f(n_to_select=2),
            transformation="keys_to_samples",
            moved_keys=["species_center"],
        )
        tr.fit(descriptor)
        result = tr.transform(system, calculator)
        result.keys_to_samples("species_center")
        desc = calculator.compute(system)
        desc.keys_to_samples("species_center")
        blocks = []
        for _, block in desc:
            fps = FPS_f(n_to_select=2)
            mask = fps.fit(block.values).get_support()
            selected_properties = block.properties[mask]
            blocks.append(
                TensorBlock(
                    values=block.values[:, mask],
                    samples=block.samples,
                    components=block.components,
                    properties=selected_properties,
                )
            )
        selected_desc = TensorMap(desc.keys, blocks)
        for i in range(len(selected_desc.keys)):
            self.assertTrue(
                np.array_equal(selected_desc.block(i).values, result.block(i).values)
            )

    def test_keys_to_properties(self):
        system = TestSystem()
        calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
        descriptor = calculator.compute(system)
        tr = Transformer(
            selector=FPS_f(n_to_select=2),
            transformation="keys_to_properties",
            moved_keys=["species_center"],
        )
        tr.fit(descriptor)
        result = tr.transform(system, calculator)
        desc = calculator.compute(system)
        desc.keys_to_properties("species_center")
        result.keys_to_properties("species_center")
        blocks = []
        for _, block in desc:
            # create a separate FPS selector for each block
            fps = FPS_f(n_to_select=2)
            mask = fps.fit(block.values).get_support()
            selected_properties = block.properties[mask]
            # put the selected features in a format rascaline can use
            blocks.append(
                TensorBlock(
                    # values, samples and component carry no information here
                    values=block.values[:, mask],
                    samples=block.samples,
                    components=block.components,
                    properties=selected_properties,
                )
            )
        selected_desc = TensorMap(desc.keys, blocks)
        for i in range(len(selected_desc.keys)):
            self.assertTrue(
                np.array_equal(selected_desc.block(i).values, result.block(i).values)
            )

    # This test uses a function which is not yet implemented in rascaline,
    # so it is temporarily commented out
    # def test_keys_to_properties_labels(self):
    #     system = TestSystem()
    #     lab = Labels(
    #         names=['species_center'],
    #         values=np.array([[0], [1]])
    #     )
    #     calculator = DummyCalculator(cutoff=3.2, delta=2, name="")
    #     descriptor = calculator.compute(system)
    #     tr = Transformer(selector = FPS_f(n_to_select=2),
    #                      transformation='keys_to_properties',
    #                      moved_keys=lab)
    #     tr.fit(descriptor)
    #     result = tr.transform(system, calculator)
    #     desc = calculator.compute(system)
    #     desc.keys_to_properties(lab)
    #     result.keys_to_properties(lab)
    #     blocks=[]
    #     for _, block in desc:
    #         # create a separate FPS selector for each block
    #         fps = FPS_f(n_to_select=2)
    #         mask = fps.fit(block.values).get_support()
    #         selected_properties = block.properties[mask]
    #         # put the selected features in a format rascaline can use
    #         blocks.append(
    #          TensorBlock(
    #              # values, samples and component carry no information here
    #              values=block.values[:, mask],
    #              samples=block.samples,
    #              components=block.components,
    #              properties=selected_properties,
    #          )
    #         )
    #     selected_desc = TensorMap(desc.keys, blocks)
    #     print(selected_desc.block(0).values, result.block(0).values)
    #     for i in range(len(selected_desc.keys)):
    #         self.assertTrue(np.array_equal(selected_desc.block(i).values,
    #                                        result.block(i).values))
    #


if __name__ == "__main__":
    unittest.main()
