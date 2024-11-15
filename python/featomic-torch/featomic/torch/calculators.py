import importlib
import sys

import featomic.calculators

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
# This module tries to re-use code from `featomic.calculators`, which contains a more
# user-friendly interface to the different calculator. At the C-API level calculators
# are just defined by a name & JSON parameter string. `featomic.calculators` defines
# one class for each name and set the `__init__` parameters with the top-level keys of
# the JSON parameters.
#
# To achieve this, we import the module in a special mode with `importlib`, defining a
# global variable `CalculatorBase` which is pointing to `CalculatorModule`. Then,
# `featomic.calculators` checks if `CalculatorBase` is defined and otherwise imports it
# from `featomic.calculator_base`.
#
# This means the same code is used to define two versions of each class: one will be
# used in `featomic` and have a base class of `featomic.CalculatorBase`, and one in
# `featomic.torch` with base classes `featomic.torch.CalculatorModule` and
# `torch.nn.Module`.


spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "featomic.torch.calculators",
    # using the code from there
    featomic.calculators.__file__,
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module

module.__dict__["CalculatorBase"] = CalculatorModule

spec.loader.exec_module(module)
