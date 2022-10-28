.. _userdoc-how-to-computing-soap:

Computing SOAP features
=======================

This examples shows how to compute the SOAP power spectrum descriptor
for each atom in each structure of a provided structure file.
The path to the structure file is taken from the first command line argument.

In the end, the descriptor is transformed
in a way compatible with how most classic machine learning (such as PCA or
linear regression) work.

The workflow is the same for every provided descriptor. Take a look at the
:ref:`userdoc-references` for a list with all descriptors and their
specific parameters.

.. tabs::

    .. group-tab:: Python

        .. literalinclude:: ../../../python/examples/compute-soap.py
            :language: python

    .. group-tab:: Rust

        .. literalinclude:: ../../../rascaline/examples/compute-soap.rs
            :language: rust

    .. group-tab:: C++

        .. literalinclude:: ../../../rascaline-c-api/examples/compute-soap.cpp
            :language: c++

    .. group-tab:: C

        .. literalinclude:: ../../../rascaline-c-api/examples/compute-soap.c
            :language: c
