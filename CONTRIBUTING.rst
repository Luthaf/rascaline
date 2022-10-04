.. _dev-contributing:

Contributing
============

By contributing to rascaline, you accept and agree to the following terms and
conditions for your present and future contributions submitted to rascaline.
Except for the license granted herein to rascaline and recipients of software
distributed by rascaline, you reserve all right, title, and interest in and to
your contributions.

Code of Conduct
---------------

As contributors and maintainers of rascaline, we pledge to respect all people
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
open an issue to discuss your changes. Use the only `master` branch 
for submitting your requests.

.. _`Github` : https://github.com/Luthaf/rascaline

Required tools
--------------

You will need to install and get familiar with the following tools when working
on rascaline:

- **git**: the software we use for version control of the source code. See
  https://git-scm.com/downloads for installation instructions.
- **the rust compiler**: you will need both ``rustc`` (the compiler) and
  ``cargo`` (associated build tool). You can install both using `rustup`_, or
  use a version provided by your operating system. We need at least Rust version
  1.56.0 to build rascaline.
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
.. _Tox: https://tox.readthedocs.io/en/latest

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

Continuous Integration pipeline is based on `cargo`_.
You can run all tests by

.. code-block:: bash

    cd <path/to/rascaline/repo>
    cargo test  # or cargo test --release to run tests in release mode

These are exactly the same tests that will be performed online in our
Github CI workflows.
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

Also, you can run individual python environment tets using `Tox`_ 
if you wish to test only specific functionalities, for example:

.. code-block:: bash

    tox -e lint  # code style
    tox -e all-deps  # python tests with all dependencies
    tox -e min-deps  # python tests with minimal dependencies
    tox -e examples  # python tests of examples
    tox -e build  # python packaging
    tox -e format  # format all files

The latter command `tox -e format` will use tox to do actual formatting instead
of just testing it.

.. _`cargo` : https://doc.rust-lang.org/cargo/

Writing your own calculator
---------------------------

For adding a new calculator take a look at the Tutorial for 
`Adding a new calculator`_.

.. _Adding a new calculator: https://luthaf.fr/rascaline/latest/devdoc/new-calculator

Contributing to the documentation
---------------------------------

The documentation of rascaline is written in reStructuredText (rst)
and uses `sphinx`_ documentation generator. In order to modify the
documentation, first create a local version on your machine as described above.
Then, build the documentation from the ``rascaline/docs`` folder:

.. code-block:: bash

    cd docs
    pip3 install -r requirements.txt
    make html

Then, visualise the local documentation
with your favourite browser (here Mozilla Firefox is used)

.. code-block:: bash

    firefox build/html/index.html

**Note**: If you write python documentation make sure that `rascaline` is not installed
in your current environment. Otherwise changes to Python docstrings might 
not be reflected in the build documentation because sphinx 
prefers the installed version compared to the one from the repo.

.. _`sphinx` : https://www.sphinx-doc.org/en/master/