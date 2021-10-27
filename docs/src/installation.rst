Installation
============

You can install rascaline in different ways depending on which language you plan
to use it from. In all cases you will need a Rust compiler, which you can
install using `rustup <https://rustup.rs/>`_ or your OS package manager.

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
    cmake <CMAKE_OPTIONS_HERE> ..
    make install

The build and installation can be configures with a few cmake options, using
``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the cmake GUI
(``cmake-gui`` or ``ccmake``). Here are the main configuration options:

+--------------------------+--------------------------------------------------------------------------------------+----------------+
| Option                   | Description                                                                          | Default        |
+==========================+======================================================================================+================+
| CMAKE_BUILD_TYPE         | Type of build: debug or release                                                      | release        |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX     | Prefix in which the library will be installed                                        | ``/usr/local`` |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR      | Path relative to ``CMAKE_INSTALL_PREFIX`` where the headers will be installed        | ``include``    |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| LIB_INSTALL_DIR          | Path relative to ``CMAKE_INSTALL_PREFIX`` where the shared library will be installed | ``lib``        |
+--------------------------+--------------------------------------------------------------------------------------+----------------+
| RASCAL_DISABLE_CHEMFILES | Disable the usage of chemfiles for reading structures from files                     | OFF            |
+--------------------------+--------------------------------------------------------------------------------------+----------------+


Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    rascaline = {git = "https://github.com/Luthaf/rascaline"}

Rascaline has one optional dependency (chemfiles), which is enabled by default.
If you want to disable it, you can use:

.. code-block:: toml

    [dependencies]
    rascaline = {git = "https://github.com/Luthaf/rascaline", default-features = false}
