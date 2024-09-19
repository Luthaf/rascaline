.. _python-utils-splines:


Splined radial integrals
========================

Classes for generating splines which can be used as tabulated radial integrals
in the various SOAP and LODE calculators.

.. All classes are based on :py:class:`rascaline.utils.RadialIntegralSplinerBase`.
.. We provides several ways to compute a radial integral: you may chose and
.. initialize a pre defined atomic density and radial basis and provide them to
.. :py:class:`rascaline.utils.SoapSpliner` or
.. :py:class:`rascaline.utils.LodeSpliner`. Both classes require `scipy`_ to be
.. installed in order to perform the numerical integrals.


.. autoclass:: rascaline.utils.SoapSpliner
    :members:
    :show-inheritance:

.. autoclass:: rascaline.utils.LodeSpliner
    :members:
    :show-inheritance:


.. _`scipy`: https://scipy.org
