.. _c-api-reference:

C API reference
===============

Rascaline offers a C API that can be called from any language able to call C
functions (in particular, this includes Python, Fortran with ``iso_c_env``, C++,
and most languages used nowadays). Convenient wrappers of the C API are also
provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust, in the ``rascaline-c-api`` crate. You can use
these functions in your own code by :ref:`installing the corresponding shared
library and header <install-c-lib>`, and then including ``rascaline.h`` and
linking with ``-lrascaline``. Alternatively, we provide a cmake package config
file, allowing you to do use rascaline like this (after installation):

.. code-block:: cmake

    find_package(rascaline)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Link to rascaline, this makes the header accessible
    target_link_libraries(MyExecutable rascaline)

The functions and types provided in ``rascaline.h`` can be grouped in three main groups:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    misc
