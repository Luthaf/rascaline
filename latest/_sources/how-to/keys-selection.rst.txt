Keys selection
==============

This examples shows how to tell rascaline to compute a set of blocks which
correspond to predefined keys. This can be used to either reduce the set of
computed blocks if we are only interested in some of them (skipping the
calculation of the other blocks); or when we need to explicitly add some blocks
to the resulting descriptor (for example to match the block set of an already
trained machine learning model).

This example uses functions discussed in :ref:`userdoc-how-to-computing-soap`
and :ref:`userdoc-how-to-property-selection`, so we suggest that you read these
first.

You can obtain a testing dataset from our :download:`website <../../static/dataset.xyz>`.

.. tabs::

    .. group-tab:: Python

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code for this example: keys-selection.py <../examples/keys-selection.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook for this example: keys-selection.ipynb <../examples/keys-selection.ipynb>`

        .. include:: ../examples/keys-selection.rst
            :start-after: start-body
            :end-before: end-body

    .. group-tab:: Rust

        To be done

    .. group-tab:: C++

        To be done

    .. group-tab:: C

        To be done
