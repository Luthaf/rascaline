Dealing with calculators
========================

.. doxygentypedef:: featomic_calculator_t

The following functions operate on :c:type:`featomic_calculator_t`:

- :c:func:`featomic_calculator`: create new calculators
- :c:func:`featomic_calculator_free`: free allocated calculators
- :c:func:`featomic_calculator_compute`: run the actual calculation
- :c:func:`featomic_calculator_name` get the name of a calculator
- :c:func:`featomic_calculator_parameters`: get the hyper-parameters of a calculator
- :c:func:`featomic_calculator_cutoffs`: get the cutoffs of a calculator

---------------------------------------------------------------------

.. doxygenfunction:: featomic_calculator

.. doxygenfunction:: featomic_calculator_free

.. doxygenfunction:: featomic_calculator_compute

.. doxygenfunction:: featomic_calculator_name

.. doxygenfunction:: featomic_calculator_parameters

.. doxygenfunction:: featomic_calculator_cutoffs

---------------------------------------------------------------------

.. doxygenstruct:: featomic_calculation_options_t
    :members:

.. doxygenstruct:: featomic_labels_selection_t
    :members:
