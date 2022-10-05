Rascaline
=========

|test| |docs|

.. inclusion-readme-intro-start

Rascaline is work in progress. Please don't use for anything important.

Rascaline is a library for computing representations for atomistic
machine learning. The core of the library is written in Rust and we provide 
APIs for C/C++ and Python as well.

List of representations
########################

.. inclusion-marker-modules-start

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Representations Name
     - Description

   * - Spherical Expansion
     - Core of representations in the SOAP (Smooth Overlap of Atomic Positions)
   * - Soap radial spectrum
     - Each atom is represenetd by the radial average of the density of its neighbors
   * - Soap power spectrum
     - Each sample represents rotationally-averaged atomic density correlations,
       built on top of the spherical expansion
   * - Sorted distances
     - Each atomic center is represented by a vector of distance to its
       neighbors within the spherical cutoff

.. inclusion-readme-intro-end

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://luthaf.fr/rascaline/index.html

.. |test| image:: https://github.com/Luthaf/rascaline/actions/workflows/tests.yml/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/Luthaf/rascaline/actions/workflows/tests.yml

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://luthaf.fr/rascaline/index.html
