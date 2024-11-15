.. _c-api-reference:

C API reference
===============

Featomic offers a C API that can be called from any language able to call C
functions (in particular, this includes Python, Fortran with ``iso_c_env``, C++,
and most languages used nowadays). Convenient wrappers of the C API are also
provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust, in the ``featomic-c-api`` crate. You can use
these functions in your own code by :ref:`installing the corresponding shared
library and header <install-c-lib>`, and then including ``featomic.h`` and
linking with ``-lfeatomic``. Alternatively, we provide a cmake package config
file, allowing you to do use featomic like this (after installation):

.. code-block:: cmake

    find_package(featomic)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Link to featomic, this makes the header accessible
    target_link_libraries(MyExecutable featomic)

The functions and types provided in ``featomic.h`` can be grouped in three main groups:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    misc
