Rascaline
=========

|test| |docs| |cov|

Rascaline is a library for the efficient computing of representations for atomistic
machine learning also called "descriptors" or "fingerprints". These representations
can be used for atomistic machine learning (ml) models including ml potentials,
visualization or similarity analysis.

The core of the library is written in Rust and we provide
APIs for C/C++ and Python as well.

.. warning::

    **Rascaline is still as the proof of concept stage. You should not use it for
    anything important.**

List of implemented representations
###################################

.. inclusion-marker-representations-start

.. list-table::
   :widths: 25 50 20
   :header-rows: 1

   * - representation
     - description
     - gradients

   * - Spherical expansion
     - Atoms are represented by the expansion of their neighbor's density on
       radial basis and spherical harmonics. This is the core of representations
       in SOAP (Smooth Overlap of Atomic Positions)
     - positions, strain, cell
   * - SOAP radial spectrum
     - Atoms are represented by 2-body correlations of their neighbors' density
     - positions, strain, cell
   * - SOAP power spectrum
     - Atoms are represented by 3-body correlations of their neighbors' density
     - positions, strain, cell
   * - LODE Spherical Expansion
     - Core of representations in LODE (Long distance equivariant)
     - positions
   * - Sorted distances
     - Each atom is represented by a vector of distance to its neighbors within
       the spherical cutoff
     - no
   * - Neighbor List
     - Each pair is represented by the vector between the atoms. This is
       intended to be used as a starting point for more complex representations
     - positions
   * - AtomicComposition
     - Obtaining the stoichiometric information of a system
     - positions, strain, cell

.. inclusion-marker-representations-end

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://luthaf.fr/rascaline/index.html

.. |test| image:: https://img.shields.io/github/check-runs/Luthaf/rascaline/master?logo=github&label=tests
    :alt: Tests status
    :target: https://github.com/Luthaf/rascaline/actions?query=branch%3Amaster

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
    :alt: Documentation
    :target: `documentation`_

.. |cov| image:: https://codecov.io/gh/Luthaf/rascaline/branch/master/graph/badge.svg
    :alt: Coverage Status
    :target: https://codecov.io/gh/Luthaf/rascaline
