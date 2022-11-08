Property Selection
==================

This examples shows how to only compute a subset of properties for each sample
with a given rascaline representation. In particular, we will use the SOAP
power spectrum representation, and select the most significant features within
a single block using farthest point sampling (FPS). We will run the calculation
for all atoms in a structure file, the path to which should be given as the
first command line argument.

This is useful if we are interested in the contribution of individual features
to the result, or if we want to reduce the computational cost by using only
part of the features for our model.

The first part of this example repeats the :ref:`userdoc-how-to-computing-soap`,
so we suggest that you read it initially.

We will use the implementation of Farthest Point Sampling from scikit-cosmo,
if you want to learn more have a look at the
`scikit-cosmo documentation <https://scikit-cosmo.readthedocs.io/en/latest/>`_.

You can obtain a testing dataset from our :download:`website <../../static/dataset.xyz>`.

.. tabs::

    .. group-tab:: Python

        .. include:: ../examples/property-selection.rst
            :start-after: start-body
            :end-before: end-body

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code: property-selection.py <../examples/property-selection.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook: property-selection.ipynb <../examples/property-selection.ipynb>`
