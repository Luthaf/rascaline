By contributing to featomic, you accept and agree to the following terms and
conditions for your present and future contributions submitted to featomic.
Except for the license granted herein to featomic and recipients of software
distributed by featomic, you reserve all right, title, and interest in and to
your contributions.

Code of Conduct
---------------

As contributors and maintainers of featomic, we pledge to respect all people
who contribute through reporting issues, posting feature requests, updating
documentation, submitting merge requests or patches, and other activities.

We are committed to making participation in this project a harassment-free
experience for everyone, regardless of level of experience, gender, gender
identity and expression, sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, or religion.

Examples of unacceptable behavior by participants include the use of sexual
language or imagery, derogatory comments or personal attacks, trolling, public
or private harassment, insults, or other unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct. Project maintainers who do not follow the
Code of Conduct may be removed from the project team.

This code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

.. Instances of abusive, harassing, or otherwise unacceptable behavior can be
.. reported by emailing xxx@xxx.org.

This Code of Conduct is adapted from the `Contributor Covenant`_, version 1.1.0,
available at https://contributor-covenant.org/version/1/1/0/

.. _`Contributor Covenant` : https://contributor-covenant.org

Getting involved
----------------

Contribution via merge requests are always welcome. Source code is
available from `Github`_. Before submitting a merge request, please
open an issue to discuss your changes. Use the only `main` branch
for submitting your requests.

.. _`Github` : https://github.com/metatensor/featomic

Required tools
--------------

You will need to install and get familiar with the following tools when working
on featomic:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **the rust compiler**: you will need both ``rustc`` (the compiler) and
  ``cargo`` (associated build tool). You can install both using `rustup`_, or
  use a version provided by your operating system. We need at least Rust version
  1.74 to build featomic.
- **Python**: you can install ``Python`` and ``pip`` from your operating system.
  We require a Python version of at least 3.6.
- **tox**: a Python test runner, cf https://tox.readthedocs.io/en/latest/. You
  can install tox with ``pip install tox``.

Additionally, you will need to install the following software, but you should
not have to interact with them directly:

- **cmake**: we need a cmake version of at least 3.10.
- **a C++ compiler** we need a compiler supporting C++11. GCC >= 5, clang >= 3.7
  and MSVC >= 15 should all work, although MSVC has not been tested yet.

.. _rustup: https://rustup.rs
.. _tox: https://tox.readthedocs.io/en/latest

Getting the code
----------------

The first step when developing featomic is to `create a fork`_ of the main
repository on github, and then clone it locally:

.. code-block:: bash

    git clone <insert/your/fork/url/here>
    cd featomic

    # setup the local repository so that the main branch tracks changes in
    # the original repository
    git remote add upstream https://github.com/metatensor/featomic/
    git fetch upstream
    git branch main --set-upstream-to=upstream/main

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

Continuous Integration pipeline is based on `cargo`_.
You can run all tests by

.. code-block:: bash

    cd <path/to/featomic/repo>
    cargo test  # or cargo test --release to run tests in release mode

These are exactly the same tests that will be performed online in our
Github CI workflows.
You can also run only a subset of tests with one of these commands:

- ``cargo test`` runs everything
- ``cargo test --package=featomic`` to run the calculators tests;
- ``cargo test --package=featomic-c-api`` to run the C/C++ tests only;

  - ``cargo test --test=run-cxx-tests`` will run the unit tests for the C/C++
    API. If `valgrind`_ is installed, it will be used to check for memory
    errors. You can disable this by setting the `FEATOMIC_DISABLE_VALGRIND`
    environment variable to 1 (`export FEATOMIC_DISABLE_VALGRIND=1` for most
    Linux/macOS shells);
  - ``cargo test --test=check-cxx-install`` will build the C/C++ interfaces,
    install them and the associated CMake files and then try to build a basic
    project depending on this interface with CMake;

- ``cargo test --package=featomic-torch`` to run the C++ TorchScript extension
  tests only;

  - ``cargo test --test=run-torch-tests`` will run the unit tests for the
    TorchScript C++ extension;
  - ``cargo test --test=check-cxx-install`` will build the C++ TorchScript
    extension, install it and then try to build a basic project depending on
    this extension with CMake;

- ``cargo test --package=featomic-python`` (or ``tox`` directly, see below) to
  run Python tests only;
- ``cargo test --lib`` to run unit tests;
- ``cargo test --doc`` to run documentation tests;
- ``cargo bench --test`` compiles and run the benchmarks once, to quickly ensure
  they still work.

You can add some flags to any of above commands to further refine which tests
should run:

- ``--release`` to run tests in release mode (default is to run tests in debug mode)
- ``-- <filter>`` to only run tests whose name contains filter, for example
  ``cargo test -- spherical_harmonics``

Also, you can run individual python tests using `tox`_
if you wish to test only specific functionalities, for example:

.. code-block:: bash

    tox -e lint  # code style
    tox -e all-deps  # python tests with all dependencies
    tox -e min-deps  # python tests with minimal dependencies
    tox -e examples  # python tests of examples
    tox -e build  # python packaging
    tox -e format  # format all files

The latter command ``tox -e format`` will use tox to do actual formatting instead
of just testing it.

.. _`cargo` : https://doc.rust-lang.org/cargo/
.. _valgrind: https://valgrind.org/

Inspecting Python code coverage
-------------------------------

The code coverage is reported at `codecov`_. You can also inspect the coverage locally.
To get the full coverage first combine all reports and open produced html file in a
browser

.. code-block:: bash

    tox
    coverage combine --append \
        ./.coverage \
        ./python/featomic-torch/.coverage
    coverage html
    firefox htmlcov/index.html

.. _codecov: https://codecov.io/gh/lab-cosmo/metatensor

Writing your own calculator
---------------------------

For adding a new calculator take a look at the tutorial for
`adding a new calculator`_.

.. _adding a new calculator: https://metatensor.github.io/featomic/latest/devdoc/how-to/new-calculator.html

Contributing to the documentation
---------------------------------

The documentation of featomic is written in reStructuredText (rst)
and uses `sphinx`_ documentation generator. In order to modify the
documentation, first create a local version on your machine as described above.
Then, build the documentation:

.. code-block:: bash

    tox -e docs

You can then visualise the local documentation
with your favourite browser (here Mozilla Firefox is used)

.. code-block:: bash

    firefox docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org/en/master/
