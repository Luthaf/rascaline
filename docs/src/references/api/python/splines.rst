.. _python-utils-splines:


Splined radial integrals
========================

The classes presented here can take arbitrary radial basis function and density;
and compute the radial integral that enters many density-based representations
such as SOAP and LODE. This enables using arbitrary, user-defined basis
functions and density with the existing calculators. Both classes require
`scipy`_ to be installed in order to perform the numerical integrals.


.. autoclass:: rascaline.splines.SoapSpliner
    :members:

.. autoclass:: rascaline.splines.LodeSpliner
    :members:


.. _`scipy`: https://scipy.org
