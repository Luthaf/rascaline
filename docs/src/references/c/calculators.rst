Dealing with calculators
========================

.. doxygentypedef:: rascal_calculator_t

The following functions operate on :c:type:`rascal_calculator_t`:

- :c:func:`rascal_calculator`: create new calculators
- :c:func:`rascal_calculator_free`: free allocated calculators
- :c:func:`rascal_calculator_compute`: run the actual calculation
- :c:func:`rascal_calculator_name` get the name of a calculator
- :c:func:`rascal_calculator_parameters`: get the hyper-parameters of a calculator

---------------------------------------------------------------------

.. doxygenfunction:: rascal_calculator

.. doxygenfunction:: rascal_calculator_free

.. doxygenfunction:: rascal_calculator_compute

.. doxygenfunction:: rascal_calculator_name

.. doxygenfunction:: rascal_calculator_parameters

---------------------------------------------------------------------

.. doxygenstruct:: rascal_calculation_options_t
    :members:

.. doxygenstruct:: rascal_labels_selection_t
    :members:
