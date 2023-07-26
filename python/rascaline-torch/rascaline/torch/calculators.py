import importlib
import sys

import rascaline.calculators

from .calculator_base import CalculatorModule


#                       CAREFUL ADVENTURER, HERE BE DRAGONS!
#
#                                         \||/
#                                         |  @___oo
#                               /\  /\   / (__,,,,|
#                              ) /^\) ^\/ _)
#                              )   /^\/   _)
#                              )   _ /  / _)
#                          /\  )/\/ ||  | )_)
#                         <  >      |(,,) )__)
#                          ||      /    \)___)\
#                          | \____(      )___) )___
#                           \______(_______;;; __;;;
#
#
# This module tries to re-use code from `rascaline.calculators`, which contains a more
# user-friendly interface to the different calculator. At the C-API level calculators
# are just defined by a name & JSON parameter string. `rascaline.calculators` defines
# one class for each name and set the `__init__` parameters with the top-level keys of
# the JSON parameters.
#
# To achieve this, we import the module in a special mode with `importlib`, defining a
# global variable `CalculatorBase` which is pointing to `CalculatorModule`. Then,
# `rascaline.calculators` checks if `CalculatorBase` is defined and otherwise imports it
# from `rascaline.calculator_base`.
#
# This means the same code is used to define two versions of each class: one will be
# used in `rascaline` and have a base class of `rascaline.CalculatorBase`, and one in
# `rascaline.torch` with base classes `rascaline.torch.CalculatorModule` and
# `torch.nn.Module`.


spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "rascaline.torch.calculators",
    # using the code from there
    rascaline.calculators.__file__,
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module

module.__dict__["CalculatorBase"] = CalculatorModule

spec.loader.exec_module(module)

# don't forget to also update `rascaline/__init__.py` and `rascaline/torch/__init__.py`
# when modifying this file
AtomicComposition = module.AtomicComposition
DummyCalculator = module.DummyCalculator
LodeSphericalExpansion = module.LodeSphericalExpansion
NeighborList = module.NeighborList
SoapPowerSpectrum = module.SoapPowerSpectrum
SoapRadialSpectrum = module.SoapRadialSpectrum
SortedDistances = module.SortedDistances
SphericalExpansion = module.SphericalExpansion
SphericalExpansionByPair = module.SphericalExpansionByPair
