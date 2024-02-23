.. _userdoc-how-to-computing-soap:

Computing SOAP features
=======================

This examples shows how to compute the SOAP power spectrum descriptor for each
atom in each system of a provided systems file. The path to the systems file is
taken from the first command line argument.

In the end, the descriptor is transformed in a way compatible with how most
classic machine learning (such as PCA or linear regression) work.

The workflow is the same for every provided descriptor. Take a look at the
:ref:`userdoc-references` for a list with all descriptors and their specific
parameters.

You can obtain a testing dataset from our :download:`website <../../static/dataset.xyz>`.

.. tabs::

    .. group-tab:: Python

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code for this example: compute-soap.py <../examples/compute-soap.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook for this example: compute-soap.ipynb <../examples/compute-soap.ipynb>`

        .. include:: ../examples/compute-soap.rst
            :start-after: start-body
            :end-before: end-body

    .. group-tab:: Rust

        .. literalinclude:: ../../../rascaline/examples/compute-soap.rs
            :language: rust

    .. group-tab:: C++

        .. literalinclude:: ../../../rascaline-c-api/examples/compute-soap.cpp
            :language: c++

    .. group-tab:: C

        .. literalinclude:: ../../../rascaline-c-api/examples/compute-soap.c
            :language: c
