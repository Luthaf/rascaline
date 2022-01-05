Defining systems
================

There are two ways you can define systems to pass to
:c:func:`rascal_calculator_compute`: the easy way is to use
:c:func:`rascal_basic_systems_read` to read all structures defined in a file,
and run the calculation on all these structures. The more complex but also more
flexible way is to create a :c:struct:`rascal_system_t` manually, implementing
all required functions; and then passing one or more systems to
:c:func:`rascal_calculator_compute`.

.. doxygenstruct:: rascal_system_t
    :members:

.. doxygenstruct:: rascal_pair_t
    :members:

---------------------------------------------------------------------

.. doxygenfunction:: rascal_basic_systems_read

.. doxygenfunction:: rascal_basic_systems_free
