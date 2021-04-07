Computing SOAP features
=======================

This examples takes the path to a structure files from command line arguments,
and compute the SOAP power spectrum descriptor for each atom in each structure.
Finally, the descriptor is transformed in a way compatible with how most classic
machine learning (such as PCA or linear regression) work.

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
