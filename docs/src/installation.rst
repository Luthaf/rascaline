Installation
============

You can install rascaline in different ways depending on which language you plan
to use it from.

.. warning::

    Rascaline is still as the proof of concept stage. You should not use it for
    anything important.

.. _install-python-lib:

Installing the Python module
----------------------------

.. code-block:: bash

    pip install git+https://github.com/Luthaf/rascaline.git

.. _install-c-lib:

Installing the C/C++ library
----------------------------

This installs a C-compatible shared library that can also be called from C++, as
well as CMake files that can be used with ``find_package(rascaline)``.

.. code-block:: bash

    git clone https://github.com/Luthaf/rascaline
    cd rascaline/rascaline-c-api
    mkdir build
    cd build
    cmake ..
    make install


Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    rascaline = {git = "https://github.com/Luthaf/rascaline"}
