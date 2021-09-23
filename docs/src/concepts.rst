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

Descriptor: data storage for calculation results
------------------------------------------------

After using a calculator on one or multiple systems, users will get the
numerical representation of their atomic systems in a `descriptor` object. This
object contains two main data arrays, and three metadata arrays. The first data
array contains the ``values`` of the representation, and is a two dimensional
array with different samples (different atomic environment/structures/etc.)
represented as rows; and different features represented as columns.

Each row is further described in the ``samples`` metadata array. This array
contains one row for each sample, fully describing this sample. Each column is
named and contains one piece of data used to identify this sample. For example,
the first row in the schematic below represent the fourth atom inside the third
structure passed to the calculator. This atom species is 8 (usually meaning
nitrogen), and this sample describe neighboring atoms of species 2.

In the same way, each column in the ``values`` array is described by one row in
the ``features`` metadata array. The exact meaning of the features depend on the
representation used, and users should refer to the corresponding documentation
for more information.

Optionally, a descriptor can also contain a ``gradients`` data array, containing
the gradients of the values with respect to relevant atomic positions. If the
gradients array is present, an additional metadata array describes each row in
the gradients, in the same way the ``samples`` metadata describes rows in
``values``. The columns of the gradients are described by the same ``features``
metadata array as the values.

.. figure:: ../static/images/descriptor.*
    :width: 80%
    :align: center

    Graphical representation of all data and metadata stored in a descriptor.
