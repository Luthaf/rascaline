.. _userdoc-how-to-long-range-descriptor:

Long-range only LODE descriptor
===============================

The :py:class:`LodeSphericalExpansion <rascaline.LodeSphericalExpansion>` allows the
calculation of a descriptor that includes all atoms within the system and projects them
onto a spherical expansion/ fingerprint within a given ``cutoff``. This is very useful
if long-range interactions between atoms are important to describe the physics and
chemistry of a collection of atoms. However, as stated the descriptor contains **ALL**
atoms of the system and sometimes it can be desired to only have a long-range/exterior
only descriptor that only includes the atoms outside a given cutoff. Sometimes there
descriptors are also denoted by far-field descriptors.

A long range only descriptor can be particular useful when one already has a good
descriptor for the short-range density like (SOAP) and the long-range descriptor (far
field) should contain different information from what the short-range descriptor already
offers.

Such descriptor can be constructed within `rascaline` as sketched by the image below.

.. figure:: ../../static/images/long-range-descriptor.*
    :align: center

In this example will construct such a descriptor using the :ref:`radial integral
splining <python-utils-splines>` tools of `rascaline`.

.. tabs::

    .. group-tab:: Python

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code for this example: long-range-descriptor.py <../examples/long-range-descriptor.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook for this example: long-range-descriptor.ipynb <../examples/long-range-descriptor.ipynb>`

        .. include:: ../examples/long-range-descriptor.rst
            :start-after: start-body
            :end-before: end-body
