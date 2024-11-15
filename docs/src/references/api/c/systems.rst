Defining systems
================

There are two ways you can define systems to pass to
:c:func:`featomic_calculator_compute`: the easy way is to use
:c:func:`featomic_basic_systems_read` to read all systems defined in a file, and
run the calculation on all these systems. The more complex but also more
flexible way is to create a :c:struct:`featomic_system_t` manually, implementing
all required functions; and then passing one or more systems to
:c:func:`featomic_calculator_compute`.

.. doxygenstruct:: featomic_system_t
    :members:

.. doxygenstruct:: featomic_pair_t
    :members:

---------------------------------------------------------------------

.. doxygenfunction:: featomic_basic_systems_read

.. doxygenfunction:: featomic_basic_systems_free
