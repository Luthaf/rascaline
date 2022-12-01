# -*- coding: utf-8 -*-
import os
import unittest
import warnings

import rascaline
from rascaline.utils import convert_old_hyperparameter_names


class TestCMakePrefixPath(unittest.TestCase):
    def test_cmake_prefix_path_exists(self):
        self.assertTrue(hasattr(rascaline._c_lib, "cmake_prefix_path"))
        self.assertTrue(isinstance(rascaline._c_lib.cmake_prefix_path, str))

    def test_cmake_files_exists(self):
        cmake = os.path.join(rascaline._c_lib.cmake_prefix_path, "rascaline")
        self.assertTrue(os.path.isfile(os.path.join(cmake, "rascaline-config.cmake")))
        self.assertTrue(
            os.path.isfile(os.path.join(cmake, "rascaline-config-version.cmake"))
        )


class TestConverter(unittest.TestCase):
    """
    Tests the hyperparameter conversions in `rascaline.utils`
    """

    def test_mode(self):
        from rascaline.utils import convert_old_hyperparameter_names

        with self.assertRaises(ValueError):
            convert_old_hyperparameter_names({}, mode="BadMode")

    def test_errant_params(self):
        with self.assertRaises(ValueError):
            convert_old_hyperparameter_names({"bad_param": 0}, mode="librascal")
        with self.assertRaises(ValueError):
            convert_old_hyperparameter_names({"bad_param": 0}, mode="dscribe")

    def test_not_gto(self):
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names(
                {"radial_basis": "NOT_GTO"}, mode="librascal"
            )
            self.assertEquals(
                str(w[-1].message),
                "WARNING: rascaline currently only supports a Gto basis.",
            )
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names({"rbf": "NOT_GTO"}, mode="dscribe")
            self.assertEquals(
                str(w[-1].message),
                "WARNING: rascaline currently only supports a Gto basis.",
            )

    def test_param_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names({"global_species": [0]}, mode="librascal")
            self.assertEquals(
                str(w[-1].message),
                "`global_species` are not required parameters in the rascaline software infrastructure",
            )
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names({"average": 0}, mode="dscribe")
            self.assertEquals(
                str(w[-1].message),
                "`average` are not required parameters in the rascaline software infrastructure",
            )
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names({"coefficient_subselection": [0]}, mode="librascal")
            self.assertEquals(
                str(w[-1].message),
                "`coefficient_subselection` are not currently supported in rascaline"
            )
        with warnings.catch_warnings(record=True) as w:
            convert_old_hyperparameter_names({"periodic": 0}, mode="dscribe")
            self.assertEquals(
                str(w[-1].message),
                "`periodic` are not currently supported in rascaline"
            )

    def test_radial_scaling(self):
        new_hypers = convert_old_hyperparameter_names(
            {
                "cutoff_function_type": "RadialScaling",
                "cutoff_function_parameters": {
                    "exponent": 3,
                    "rate": 1.0,
                    "scale": 1.5,
                },
            },
            mode="librascal",
        )
        self.assertEqual(new_hypers["radial_scaling"]['Willatt2018']["exponent"], 3)
        self.assertEqual(new_hypers["radial_scaling"]['Willatt2018']["scale"], 1.5)
        self.assertEqual(new_hypers["radial_scaling"]['Willatt2018']["rate"], 1.0)


    def test_shifted_cosine(self):
        new_hypers = convert_old_hyperparameter_names(
            {
                "cutoff_function_type": "ShiftedCosine",
                "cutoff_smooth_width": 0.5
            },
            mode="librascal",
        )
        self.assertEqual(new_hypers["cutoff_function"]['ShiftedCosine']["width"], 0.5)

if __name__ == "__main__":
    unittest.main()
