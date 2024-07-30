Sample Selection
================

This examples shows how to compute a representation only for a subset of the
available samples. In particular, we will compute the SOAP power spectrum
representation for a specific subset of atoms, out of all atoms the code reads
from a file. The path to the system file is taken from the first command line
argument.

This can be useful if we are only interested in certain systems in a large
dataset, or if we need to determine the effect of a certain type of atoms on
some system properties. In the following, we will look at the tools with which
sample selection can be done in rascaline.

The first part of this example repeats the :ref:`userdoc-how-to-computing-soap`, so we
suggest that you read it initially.

You can obtain a testing dataset from our :download:`website <../../static/dataset.xyz>`.

.. tabs::

    .. group-tab:: Python

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code for this example: sample-selection.py <../examples/sample-selection.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook for this example: sample-selection.ipynb <../examples/sample-selection.ipynb>`

        .. include:: ../examples/sample-selection.rst
            :start-after: start-body
            :end-before: end-body

    .. group-tab:: Rust

        To be done

    .. group-tab:: C++

        To be done

    .. group-tab:: C

        To be done
