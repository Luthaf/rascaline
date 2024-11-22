Code organization
-----------------

The code is organized in three main products, each in a separate directory:

- ``featomic/`` contains the main Rust implementation of all calculators, and
  the corresponding unit and regression tests;
- ``featomic-torch/`` contains the TorchScript bindings to featomic, written in
  C++;
- ``python/`` contains the Python interface to featomic and featomic-torch, and
  the corresponding tests

Finally, ``docs/`` contains the documentation for everything related to
featomic.

The main featomic crate
^^^^^^^^^^^^^^^^^^^^^^^^

Inside the main featomic crate, the following code organization is used:

- ``featomic/benches``: benchmarks of the code on some simple systems;
- ``featomic/tests``: regression tests for all calculators;
- ``featomic/src/system/``: definition of everything related to systems:
  ``UnitCell``, the ``System`` trait and ``SimpleSystem`` implementation;
- ``featomic/src/calculator.rs``: convenience wrapper around implementations of
  ``CalculatorBase`` that setup everything before a calculation;
- ``featomic/src/calculators/``: definition of the ``CalculatorBase`` trait and
  various implementations of this trait;
