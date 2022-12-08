Keys Selection
==============

This examples shows how to compute a set of blocks which correspond to p
redefined keys.

This is useful, for example, in 2 cases: when we are interested in analyzing
only some of the blocks (so there is no need to generate the rest, wasting
resources on this), and when we, on the contrary, need to explicitly add some
blocks to the resulting descriptor (to match the machine learning model, for
instance).

This example uses the functions discussed in
:ref:`userdoc-how-to-computing-soap` and
:ref:`userdoc-how-to-property-selection`, so we suggest that you read it
initially.

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
