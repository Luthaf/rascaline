.. _userdoc-calculators:

Calculator reference
--------------------

Below is a list of all calculators available. Calculators are the core of
featomic and are algorithms for transforming Cartesian coordinates into
representations suitable for machine learning. Each calculators has a different
approach of this transformation but some belong to the same family. To learn
more about these connections and the theory you may consider our
:ref:`userdoc-explanations` section.

Each calculators is registered globally with a name (specified in the
corresponding documentation page) that can be used to construct this calculator
with ``Calculator::new`` in Rust, ``featomic_calculator`` in C or
``featomic::Calculator`` in C++. The hyper-parameters of the calculator must be
given as a JSON formatted string. The possible fields in the JSON are documented
as a `JSON schema`_, and rendered in the pages below.

After the initialization of each calculator the computation of the
representations is performed using the
:py:func:`featomic.calculators.CalculatorBase.compute()` method.

.. _JSON schema: https://json-schema.org/

.. toctree::
    :maxdepth: 1

    spherical-expansion
    spherical-expansion-by-pair
    lode-spherical-expansion
    soap-radial-spectrum
    soap-power-spectrum
    atomic-composition
    neighbor-list
    sorted-distances
