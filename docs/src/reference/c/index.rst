C API reference
===============

Rascaline offers a C API (that can also be called directly from C++). The C API
is implemented in Rust, in the ``rascaline-c-api`` crate. You can use this API
by :ref:`installing the corresponding shared library and header
<install-c-lib>`, and then including ``rascaline.h`` and linking with
``-lrascaline``. Alternatively, we provide a cmake package config file, allowing
you to do use rascaline like this (after installation):

.. code-block:: cmake

    find_package(rascaline)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Link to rascaline, this makes the header accessible
    target_link_libraries(MyExecutable rascaline)

The functions and types provided in ``rascaline.h`` can be grouped in four main groups:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    descriptor
    misc
