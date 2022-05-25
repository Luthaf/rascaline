.. .. _cxx-api-reference:

.. C++ API reference
.. =================

.. Rascaline offers a C++ API, built on top of the :ref:`C API <c-api-reference>`.
.. You can the provided classes and functions in your own code by :ref:`installing
.. the corresponding shared library and header <install-c-lib>`, and then including
.. ``rascaline.hpp`` and linking with ``-lrascaline``. Alternatively, we provide a
.. cmake package config file, allowing you to do use rascaline like this (after
.. installation):

.. .. code-block:: cmake

..     find_package(rascaline)

..     # add executables/libraries
..     add_executable(MyExecutable my_sources.cxx)
..     add_library(MyLibrary my_sources.cxx)

..     # Link to rascaline, this makes the header accessible
..     target_link_libraries(MyExecutable rascaline)

.. The functions and types provided in ``rascaline.hpp`` can be grouped in three main groups:

.. .. toctree::
..     :maxdepth: 1

..     systems
..     calculators
..     misc
