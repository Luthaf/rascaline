Architecture
============

Code organization
-----------------

The code is organized in three main products, each in a separate directory:

- ``rascaline/`` contains the main Rust implementation of all calculators, and
  the corresponding unit and regression tests;
- ``rascaline-c-api/`` is a Rust crate containing the implementation of the
  rascaline C API;
- ``python/`` contains the Python interface to rascaline, and the corresponding
  tests

Finally, ``docs/`` contains the documentation for everything related to
rascaline.

The main rascaline crate
^^^^^^^^^^^^^^^^^^^^^^^^

Inside the main rascaline crate, the following code organization is used:

- ``rascaline/benches``: benchmarks of the code on some simple systems;
- ``rascaline/tests``: regression tests for all calculators;
- ``rascaline/src/system/``: definition of everything related to systems:
  ``UnitCell``, the ``System`` trait and ``SimpleSystem`` implementation;
- ``rascaline/src/calculator.rs``: convenience wrapper around implementations of
  ``CalculatorBase`` that setup everything before a calculation;
- ``rascaline/src/calculators/``: definition of the ``CalculatorBase`` trait and
  various implementations of this trait;
