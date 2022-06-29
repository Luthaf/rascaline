.. _calculators-list:

Implemented calculators
=======================

Below is a list of all calculators available in rascaline. Each calculators is
registered globally with a name (specified in the corresponding documentation
page) that can be used to construct this calculator with ``Calculator::new`` in
Rust, `rascal_calculator` in C or C++. The hyper-parameters of the calculator
must be given as a JSON formatted string. The possible fields in the JSON are
documented as a `JSON schema`_, and rendered in the pages below.

.. _JSON schema: https://json-schema.org/

.. toctree::
    :maxdepth: 1

    spherical-expansion
    soap-radial-spectrum
    soap-power-spectrum
    sorted-distances
