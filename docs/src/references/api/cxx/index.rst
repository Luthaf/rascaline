.. _cxx-api-reference:

C++ API reference
=================

Featomic offers a C++ API, built on top of the :ref:`C API <c-api-reference>`.
You can the provided classes and functions in your own code by :ref:`installing
the corresponding shared library and header <install-c-lib>`, and then including
``featomic.hpp`` and linking with ``-lfeatomic``. Alternatively, we provide a
cmake package config file, allowing you to do use featomic like this (after
installation):

.. code-block:: cmake

    find_package(featomic)

    # add executables/libraries
    add_executable(MyExecutable my_sources.cxx)
    add_library(MyLibrary my_sources.cxx)

    # Link to featomic, this makes the header accessible
    target_link_libraries(MyExecutable featomic)

The functions and types provided in ``featomic.hpp`` can be grouped in three main groups:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    misc
