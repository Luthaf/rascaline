.. _dev-getting-started:

Getting started
===============

Required tools
--------------

You will need to install and get familiar with the following tools when working
on rascaline:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **the rust compiler**: you will need both ``rustc`` (the compiler) and
  ``cargo`` (associated build tool). You can install both using `rustup`_, or
  use a version provided by your operating system. We need at least Rust version
  1.42.0 to build rascaline.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.6.
- **tox**: a Python test runner, cf https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Additionally, you will need to install the following software, but you should
not have to interact with them directly:

- **cmake**: we need a cmake version of at least 3.10.
- **a C++ compiler** we need a compiler supporting C++11. GCC >= 5, clang >= 3.7
  and MSVC >= 15 should all work, although MSVC has not been tested yet.

.. _rustup: https://rustup.rs/

Getting the code
----------------

The first step when developing rascaline is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd rascaline

    # setup the local repository so that the master branch tracks changes in
    # the main repository
    git remote add upstream https://github.com/Luthaf/rascaline/
    git fetch upstream
    git branch master --set-upstream-to=upstream/master

Once you get the code locally, you will want to run the tests to check
everything is working as intended. See the next section on this subject.

If everything is working, you can create your own branches to work on your
changes:

.. code-block:: bash

    git checkout -b <my-branch-name>
    # code code code

    # push your branch to your fork
    git push -u origin <my-branch-name>
    # follow the link in the message to open a pull request (PR)

.. _create a fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo

Running tests
-------------

Once you have installed all dependencies and the cloned the repository locally,
you can run all tests with

.. code-block:: bash

    cd <path/to/rascaline/repo>
    cargo test  # or cargo test --release to run tests in release mode

You can also run only a subset of tests with one of these commands:

- ``cargo test`` runs everything
- ``cargo test --test=<test-name>`` to run only the tests in ``tests/<test-name>.rs``;
    - ``cargo test --test=python-api`` (or ``tox`` directly) to run Python tests only;
    - ``cargo test --test=c-api`` to run the C API tests only;
- ``cargo test --lib`` to run unit tests;
- ``cargo test --doc`` to run documentation tests;
- ``cargo bench --test`` compiles and run the benchmarks once, to quickly ensure
  they still work.

You can add some flags to any of above commands to further refine which tests
should run:

- ``--release`` to run tests in release mode (default is to run tests in debug mode)
- ``-- <filter>`` to only run tests whose name contains filter, for example ``cargo test -- spherical_harmonics``
- ``--package rascaline`` to run tests defined in the rascaline crate (the core implementation)
- ``--package rascaline-c-api`` to run tests defined in the rascaline-c-api
  crate (the C API implementation)
