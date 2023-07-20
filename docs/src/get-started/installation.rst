Installation
============

You can install rascaline in different ways depending on which language you plan
to use it from. In all cases you will need a Rust compiler, which you can
install using `rustup <https://rustup.rs/>`_ or your OS package manager.

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

+--------------------------------------+-----------------------------------------------+----------------+
| Option                               | Description                                   | Default        |
+======================================+===============================================+================+
| CMAKE_BUILD_TYPE                     | Type of build: debug or release               | release        |
+--------------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                 | Prefix in which the library will be installed | ``/usr/local`` |
+--------------------------------------+-----------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR                  | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``include``    |
|                                      |  where the headers will be installed          |                |
+--------------------------------------+-----------------------------------------------+----------------+
| LIB_INSTALL_DIR                      | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``lib``        |
|                                      | where the shared library will be installed    |                |
+--------------------------------------+-----------------------------------------------+----------------+
| BUILD_SHARED_LIBS                    | Default to installing and using a shared      | ON             |
|                                      | library instead of a static one               |                |
+--------------------------------------+-----------------------------------------------+----------------+
| RASCALINE_INSTALL_BOTH_STATIC_SHARED | Install both the shared and static version    | ON             |
|                                      | of the library                                |                |
+--------------------------------------+-----------------------------------------------+----------------+
| RASCALINE_ENABLE_CHEMFILES           | Enable the usage of chemfiles for reading     | ON             |
|                                      | structures from files                         |                |
+--------------------------------------+-----------------------------------------------+----------------+
| RASCALINE_FETCH_EQUISTORE            | Automatically fetch, build and install        | OFF            |
|                                      | equistore (a dependency of rascaline)         |                |
+--------------------------------------+-----------------------------------------------+----------------+

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


.. _install-torch-script:

Installing the TorchScript bindings
-----------------------------------

For usage from Python
^^^^^^^^^^^^^^^^^^^^^

Building from source:

.. code-block:: bash

    git clone https://github.com/luthaf/rascaline
    cd rascaline/python/rascaline-torch
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/luthaf/rascaline#subdirectory=python/rascaline-torch


For usage from C++
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/lab-cosmo/rascaline
    cd rascaline/rascaline-torch
    mkdir build && cd build
    cmake ..
    # configure cmake if needed
    cmake --build . --target install

Compiling the TorchScript bindings requires you to already have already
installed multiple dependencies:

- the C++ part of PyTorch, which you can install `on it's own
  <https://pytorch.org/get-started/locally/>`_, or use the installation that
  comes with a Python installation by adding the output of the command below
  to ``CMAKE_PREFIX_PATH``:

  .. code-block:: bash

    python -c "import torch; print(torch.utils.cmake_prefix_path)"

- :ref:`the C++ interface of rascaline <install-c-lib>`, which itself requires
  the `C++ interface of equistore`_;
- the `TorchScript interface of equistore`_. We can download and build an
  appropriate version of it automatically by setting the cmake option
  ``-DRASCALINE_TORCH_FETCH_EQUISTORE_TORCH=ON``

If any of these dependencies is not in a standard location, you should specify
the installation directory when configuring cmake with ``CMAKE_PREFIX_PATH``.
Other useful configuration options are:

+---------------------------------------+-----------------------------------------------+----------------+
| Option                                | Description                                   | Default        |
+=======================================+===============================================+================+
| CMAKE_BUILD_TYPE                      | Type of build: debug or release               | release        |
+---------------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                  | Prefix in which the library will be installed | ``/usr/local`` |
+---------------------------------------+-----------------------------------------------+----------------+
| CMAKE_PREFIX_PATH                     | ``;``-separated list of path where CMake will |                |
|                                       | search for dependencies.                      |                |
+---------------------------------------+-----------------------------------------------+----------------+
| RASCALINE_TORCH_FETCH_EQUISTORE_TORCH | Should CMake automatically download and       | OFF            |
|                                       | install equistore-torch?                      |                |
+---------------------------------------+-----------------------------------------------+----------------+

.. _C++ interface of equistore: https://lab-cosmo.github.io/equistore/latest/get-started/installation.html#installing-the-c-and-c-library
.. _TorchScript interface of equistore: https://lab-cosmo.github.io/equistore/latest/get-started/installation.html#for-usage-from-c
