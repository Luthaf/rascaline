.. _core-concepts:

Core concepts
=============

Rascaline is a library computing representations of atomic systems for machine
learning applications. These representations encode fundamental symmetries of
the systems to ensure that the machine learning algorithm is as efficient as
possible. Examples of representations include the `Smooth Overlap of Atomic
Positions <SOAP_>`_ (SOAP), `Behler-Parrinello symmetry functions <BPSF_>`_,
`Coulomb matrices`_, and many others. This documentation does not describe each
method in details, delegating instead to many other good resources on the
subject. This section in particular explains the three core objects rascaline is
built upon: systems, calculators and descriptors.

.. figure:: ../static/images/core-concepts.*

    Schematic representations of the three core concepts in rascaline: systems,
    calculators and descriptors. The core operation provided by this library to
    compute the representation (associated with a given calculator) of one or
    multiple systems, getting the corresponding data in a descriptor.

.. _SOAP: https://doi.org/10.1103/PhysRevB.87.184115
.. _BPSF: https://doi.org/10.1063/1.3553717
.. _Coulomb matrices: https://doi.org/10.1103/PhysRevLett.108.058301

Systems: atoms and molecules
----------------------------

Systems describe the input data rascaline uses to compute various
representations. They contains information about the atomic positions, different
atomic types, unit cell and periodicity, and are responsible for computing the
neighbors of each atomic center.

Rascaline uses systems in a generic manner, and while it provides a default
implementation called ``SimpleSystem`` it is able to use data from any source by
going through a few lines of adaptor code. This enables using it directly inside
molecular simulation engines, re-using the neighbors list calculation done by
the engine, when using machine learning force-fields in simulations.

Both implementation and data related to systems are thus provided by users of
the rascaline library.

Calculators: computing representations
--------------------------------------

Calculators are provided by rascaline, and compute a single representations.
There is a calculator for the :ref:`sorted distances vector <sorted-distances>`
representation, another one for the :ref:`spherical expansion
<spherical-expansion>` representation, and hopefully soon many others.

All calculators are registered globally in rascaline, and can be constructed
with a name and a set of parameters (often called hyper-parameters). These
parameters control the features of the final representation: how many are they,
and what do they represent. All :ref:`available calculators <calculators-list>`
and the corresponding parameters are documented.

From a user perspective, calculators are black boxes that take systems as input
and returns a descriptor object, described below.

Descriptors: data storage for atomistic machine learning
--------------------------------------------------------

After using a calculator on one or multiple systems, users will get the
numerical representation of their atomic systems in a ``descriptor`` object.
Rascaline uses `equistore`_ ``TensorMap`` type when returning descriptors.

.. _equistore: https://lab-cosmo.github.io/equistore/latest/

A ``TensorMap`` can be seen as a dictionary mapping some keys to a set of data
blocks. Each block contains both data (and gradients) arrays — i.e.
multidimensional arrays containing the descriptor values — and metadata
describing the different dimensions of these arrays. Which keys are present in a
``TensorMap`` will depend on ``Calculator`` being used. Typically,
representation using one-hot encoding of atomic species will have species keys
(for example ``species_center``, ``species_neighbor``, *etc.*), and equivariant
representations will have keys for the different equivariance classes
(``spherical_harmonics_l`` for SO(3) equivariants, *etc.*).

For more information on ``TensorMap`` and what can be done with one, please see
the `equistore`_ documentation.
