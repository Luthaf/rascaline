Rascaline
=========

|test| |docs|

Rascaline is a library for the efficient computing of representations for atomistic
machine learning also called "descriptors" or "fingerprints". These representation
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
   :widths: 25 50 10 10
   :header-rows: 1

   * - Representations Name
     - Description
     - Features
     - Gradients

   * - Spherical Expansion
     - Core of representations in SOAP (Smooth Overlap of Atomic Positions)
     - ✓
     - ✓
   * - Soap radial spectrum
     - Each atom is represenetd by the radial average of the density of its neighbors
     - ✓
     - ✓
   * - Soap power spectrum
     - Each sample represents rotationally-averaged atomic density correlations,
       built on top of the spherical expansion
     - ✓
     - ✓
   * - Sorted distances
     - Each atomic center is represented by a vector of distance to its
       neighbors within the spherical cutoff
     - ✓
     - 

.. inclusion-marker-representations-end

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://luthaf.fr/rascaline/index.html

.. |test| image:: https://github.com/Luthaf/rascaline/actions/workflows/tests.yml/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/Luthaf/rascaline/actions/workflows/tests.yml

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://luthaf.fr/rascaline/index.html
