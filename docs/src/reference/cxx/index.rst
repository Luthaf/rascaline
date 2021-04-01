C++ API reference
=================

TODO

.. code-block:: cmake

    find_package(rascaline)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Link to rascaline, this makes the header accessible
    target_link_libraries(MyExecutable rascaline)

The functions and types provided in ``rascaline.hpp`` can be grouped in four main groups:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    descriptor
    misc
