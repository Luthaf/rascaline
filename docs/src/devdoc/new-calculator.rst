Tutorial: Adding a new calculator
=================================

Introduction
------------

So you would like to add a new :ref:`calculator <calculators-list>` to rascaline?

In this tutorial, we will go over all the steps required to create a new
calculator. For simplicity sake, the calculator we will implement will be very
basic, keeping the focus on how different bits of the code interact with one
another instead of complex math or performance tricks.

The calculator that we will create computes an atom-centered representation,
where each atomic environment is represented with the moments of the positions
of the neighbors up to a maximal order. Each atomic species in the neighborhood
will be considered separately. The resulting descriptor will represent an
atom-centered environment :math:`\ket{\mathcal{A}_i}` on a basis of species
:math:`\alpha` and moment order :math:`k`:

.. math::

    \braket{\alpha k | \mathcal{A}_i} = \frac{1}{N_\text{neighbors}} \sum_{j \in \mathcal{A}_i} r_{ij}^k \ \delta_{\alpha, \alpha_j}

.. figure:: ../../static/images/moments-descriptor.*
    :width: 40%
    :align: center

Throughout this tutorial, very basic knowledge of the Rust and Python
programming languages is assumed. If you are just starting up, you may find the
official `Rust book <https://doc.rust-lang.org/stable/book/>`_ useful; as well
as the documentation for the `standard library
<https://doc.rust-lang.org/stable/std/>`_; and the `API documentation`_ for
rascaline itself.

We will also assume that you have a local copy of the rascaline git repository,
and can build the code and run the tests. If not, please look at the
corresponding :ref:`documentation <dev-getting-started>`.

.. _API documentation: ../reference/rust/rascaline/index.html

The traits we'll use
--------------------

Two of the three :ref:`core concepts <core-concepts>` in rascaline are
represented in the code as Rust traits: systems implements the `System`_ trait,
and calculators implement the `CalculatorBase`_ trait. Traits (also called
interfaces in other languages) define contracts that the implementing code must
follow, in the form of a set of function and documented behavior for these
functions. Fulfilling this contract allow to add new systems which work with all
calculators, already implement or not; and new calculators which can use any
system, already implemented or not.

In this tutorial, our goal is to write a new struct implementing
`CalculatorBase`_. This implementation will take as input a slice of boxed
`System`_ trait objects, and using data from those fill up a `TensorMap`_
(defined in the equistore crate).

.. _System: ../reference/rust/rascaline/systems/trait.System.html
.. _CalculatorBase: ../reference/rust/rascaline/calculators/trait.CalculatorBase.html
.. _Calculator: ../reference/rust/rascaline/struct.Calculator.html
.. _TensorMap: ../reference/rust/equistore/tensor/struct.TensorMap.html

Implementation
--------------

Let's start by creating a new file in ``rascaline/src/calculators/moments.rs``,
and importing everything we'll need. Everything in here will be explained when
we get to using it.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s1_scaffold.rs
   :language: rust
   :start-after: [imports]
   :end-before: [imports]

Then, we can define a struct for our new calculator ``GeometricMoments``. It
will contain two fields: ``cutoff`` to store the cutoff radius, and
``max_moment`` to store the maximal moment to compute.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s1_scaffold.rs
   :language: rust
   :start-after: [struct]
   :end-before: [struct]

We can then write a skeleton implementation for the `CalculatorBase`_ trait,
leaving all function unimplemented with the ``todo!()`` macro.
``CalculatorBase`` is the trait defining all the functions required for a
calculator. Users might be more familiar with the concrete struct `Calculator`_,
which uses a ``Box<dyn CalculatorBase>`` (i.e. a pointer to a
``CalculatorBase``) to provide its functionalities.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s1_scaffold.rs
   :language: rust
   :start-after: [impl]
   :end-before: [impl]

We'll go over these functions one by one, explaining what they do as we go. Most
of the functions here are used to communicate metadata about the calculator and
the representation, and the ``compute`` function does the main part of the work.

Calculator metadata
^^^^^^^^^^^^^^^^^^^

The first function returning metadata about the calculator itself is ``name``,
which should return a user-facing name for the current instance of the
descriptor. As a quick refresher on Rust, all functions return the last (and in
this case only) expression. Here the expression creates a reference to a str
(``&str``) and then convert it to an heap-allocated ``String`` using the `Into`_
trait.

.. _Into: https://doc.rust-lang.org/std/convert/trait.Into.html

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::name]
   :end-before: [CalculatorBase::name]
   :dedent: 4

Then, the ``get_parameters`` function should return the parameters used to
create the current instance of the calculator in JSON format. To this end, we
use `serde`_ and ``serde_json`` everywhere in rascaline, so it is a good idea to
do the same here. Let's start by adding the corresponding ``#[derive]`` to the
definition of ``GeometricMoments``, and use it to implement the function.

.. _serde: https://serde.rs/

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [struct]
   :end-before: [struct]

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::parameters]
   :end-before: [CalculatorBase::parameters]
   :dedent: 4

One interesting thing here is that ``serde_json::to_string`` returns a
``Result<String, serde::Error>``, and we use ``expect`` to extract the string
value. This `Result`_ would only contain an error if ``GeometricMoments``
contained maps with non-string keys, which is not the case here. ``expect``
allow us to indicate we don't ever expect this function to fail, but if it were
to return an error, then the code would immediately stop and show the given
message (using a `panic`_).

.. _Result: https://doc.rust-lang.org/std/result/index.html
.. _panic: https://doc.rust-lang.org/std/macro.panic.html

Representation metadata
^^^^^^^^^^^^^^^^^^^^^^^

The next set of functions in the `CalculatorBase`_ trait is used to communicate
metadata about the representation, and called by the concrete `Calculator`_
struct when initializing and allocating the corresponding memory.

Keys
++++

First, we have one function defining the set of keys that will be in the final
``TensorMap``. In our case, we will want to have the center atom species and the
neighbor atom species as keys. This allow to only store data if a given neighbor
is actually present around a given center.

We could manually create a set of `Labels`_ with a `LabelsBuilder`_ and return
them. But since multiple calculators will create the same kind of keys, there
are already implementation of typical species keys. Here we use
``CenterSingleNeighborsSpeciesKeys`` to create a set of keys containing the
center species and one neighbor species. This key builder requires a ``cutoff``
(to determine which neighbors it should use) and ``self_pairs`` indicated
whether atoms should be considered to be their own neighbor or not.

.. _Labels: ../reference/rust/equistore/labels/struct.Labels.html
.. _LabelsBuilder: ../reference/rust/equistore/labels/struct.LabelsBuilder.html

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::keys]
   :end-before: [CalculatorBase::keys]
   :dedent: 4

Samples
+++++++

Having defined the keys, we need to define the metadata associated with each
block. For each block, the first set of metadata — called the **samples** --
describes the rows of the data. Three functions are used to define the samples:
first, ``features_names`` defines the name associated with the different columns
in the sample labels. Then, ``samples`` determines the set of samples associated
with each key/block. The return type of the ``samples`` function takes some
unpacking: we are returning a `Result`_ since any call to a `System`_ function
can fail. The non-error case of the result is a ``Vec<Arc<Labels>>``: we need
one set of `Labels`_ for each key/block. Finally, the labels can be the same
between different keys, and ``Arc`` allow using the same set of labels for
different keys without duplicating memory.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::samples]
   :end-before: [CalculatorBase::samples]
   :dedent: 4

Like for ``CalculatorBase::keys``, we could manually write code to detect the
right set of samples for each key. But since a lot of representation are built
on atom-centered neighborhoods, there is already a tool to create the right set
of samples in the form of ``AtomCenteredSamples``.

Components
++++++++++

The next set of metadata associated with a block are the components. Each block
can have 0 or more components, that should be used to store metadata and
information about symmetry operations or any kind of tensorial components.

Here, we dont' have any components (the ``GeometricMoments`` representation is
invariant), so we just return a list (one for each key) of empty vectors.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::components]
   :end-before: [CalculatorBase::components]
   :dedent: 4


Properties
++++++++++

The *properties* define metadata associated with the columns of the data arrays.
Like for the samples, we have one function to define the set of names associated
with each variable in the properties `Labels`_, and one function to compute the
set of properties defined for each key.

In our case, there is only one variable in the properties labels, the power
:math:`k` used to compute the moment. When building the full list of Labels for
each key in ``CalculatorBase::properties``, we use the fact that the properties
are the same for each key/block; and return multiple references to the same
``Arc<Labels>``.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::properties]
   :end-before: [CalculatorBase::properties]
   :dedent: 4


Gradients
+++++++++

Finally, we have metadata related to the gradients. First, the
``supports_gradient`` function should return which if any of the gradients can
be computed by the current calculator. Typically ``parameter`` is either
``"positions"`` or ``"cell"``. Here we only support computing the gradients with
respect to positions.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::supports_gradient]
   :end-before: [CalculatorBase::supports_gradient]
   :dedent: 4

If the user request the calculation of some gradients, and the calculator
supports it, the next step is to define the same set of metadata as for the
values above: samples, components and properties. Properties are easy, because
they are the same between the values and the gradients. The components are also
similar, with some additional components added at the beginning depending on the
kind of gradient. For example, if a calculator uses ``[first, second]`` as it's
set of components, the ``"positions"`` gradient would use ``[direction, first,
second]``, where ``direction`` contains 3 entries (x/y/z). The ``"cell"``
gradients would use ``[direction_1, direction_2, first, second]``, with
``direction_1`` and ``direction_2`` containing 3 entries (x/y/z) each.

Finally, the samples needs to be defined. For the ``"cell"`` gradients, there is
always exactly one gradient sample per value sample. For the ``"positions"``
gradient samples, we could have one gradient sample for each atom in the same
structure for each value sample. However, this would create a very large number
of gradient samples (number of atoms squared), and a lot of entries would be
filled with zeros. Instead, each calculator which supports positions gradients
must implement the ``positions_gradient_samples`` function, and use it to return
only the sample associated with non-zero gradients. This function get as input
the set of keys, the list of samples associated with each key, and the list of
systems on which we want to run the calculation.

We are again using the ``AtomCenteredSamples`` here to share code between
multiple calculators all using atom-centered samples.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::positions_gradient_samples]
   :end-before: [CalculatorBase::positions_gradient_samples]
   :dedent: 4


We are now done defining the metadata associated with our ``GeometricMoments``
calculator! In the next section, we'll go over the actual calculation of the
representation, and how to use the functions provided by `System`_.

The compute function
^^^^^^^^^^^^^^^^^^^^

We are finally approaching the most important function in `CalculatorBase`_,
``compute``. This function takes as input a list of systems and a `TensorMap`_
in which to write the results of the calculation. The function also returns a
`Result`_, to be able to indicate that an error was reached during the
calculation.

The `TensorMap`_ is initialized by the concrete `Calculator`_ struct, according
to parameters provided by the user. In particular, the tensor map will only
contain samples and properties requested by th user, meaning that the code in
``compute`` should check for each block whether a particular sample
(respectively property) is present in ``block.samples`` (resp.
``block.property``) before computing it.

This being said, let's start writing our ``compute`` function. We'll defensively
check that the tensor map keys match what we expect from them, and return a unit
value ``()`` wrapped in ``Ok`` at the end of the function.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_1.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

From here, the easiest way to implement our geometric moments descriptor is to
iterate over the systems, and then iterate over the pairs in the system. Before
we can get the pairs with ``system.pairs()``, we need to compute the neighbors
list for our current cutoff, using ``system.compute_neighbors()``, which
requires a mutable reference to the system to be able to store the list of
computed pairs (hence the iteration using ``systems.iter_mut()``).

All the functions on the `System`_ trait return `Result`_, but in contrary to
the ``CalculatorBase::parameters`` function above, we want to send the possible
errors back to the user so that they can deal with them as they want. The
question mark ``?`` operator does exactly that: if the value returned by the
called function is ``Err(e)``, ``?`` immediately returns ``Err(e)``; and if the
result is ``Ok(v)``, ``?`` extract the ``v`` and the execution continues.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_2.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

For each pair, we now have to find the corresponding block (using the center and
neighbor species values), and check wether the corresponding sample was
requested by the user.

To find blocks and check for samples, we can use the `Labels::position`_
function on the keys and the samples `Labels`_. This function returns an
``Option<usize>``, which will be ``None`` is the label (key or sample) was not
found, and ``Some(position)`` where ``position`` is an unsigned integer if the
label was found. For the keys, we know the blocks must exists, so we again use
``expect`` to immediately extract the value of the block index and access the
block. For the samples, we keep them as ``Option<usize>`` and will deal with
missing samples later.

One thing to keep in mind is that a given pair can participate to two different
samples. If two atoms ``i`` and ``j`` are closer than the cutoff, the list of
pairs will only contain the ``i-j`` pair, and not the ``j-i`` pair (it is a
so-called half neighbors list). That being said, we can get the list of species
with ``system.species()`` before the loop over pairs, and then construct the two
candidate samples and check for their presence. If neither of the samples was
requested, then we can skip the calculation for this pair. We also use
``system.pairs_containing()`` to get the number of neighbors a given center has.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_3.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4


.. _Labels::position: ../reference/rust/equistore/labels/struct.Labels.html#method.position

Now, we can check if the samples are present, and if they are, iterate over the
requested features, compute the moments for the current pair distance, and
accumulate it in the descriptor values array:

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_4.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4


Finally, we can deal with the gradients. We first check if gradient data is
defined in the descriptor we need to fill, by checking if it is defined on the
first block (we know it is either defined on all blocks or none).

If we need to compute the gradients with respect to atomic positions, we will us
the following expression:

.. math::

    \frac{\partial}{\partial \vec{r_{j}}} \braket{\alpha k | \chi_i} = \frac{\vec{r_{ij}}}{r_{ij}} \cdot \frac{k \ r_{ij}^{k - 1} \ \delta_{\alpha, \alpha_j}}{N_\text{neighbors}} = \vec{r_{ij}} \frac{k \ r_{ij}^{k - 2} \ \delta_{\alpha, \alpha_j}}{N_\text{neighbors}}

The code to compute gradients is very similar to the code computing the
representation, checking the existence of a given gradient sample before writing
to it. There are now four possible contributions for a given pair:
:math:`\partial \ket{\chi_i} / \partial r_j`, :math:`\partial \ket{\chi_j} /
\partial r_i`, :math:`\partial \ket{\chi_i} / \partial r_i` and :math:`\partial
\ket{\chi_j} / \partial r_j`, where :math:`\ket{\chi_i}` is the representation
around atom :math:`i`. Another way to say it is that in addition to the
gradients of the descriptor centered on :math:`i` with respect to atom
:math:`j`, we also need to account for the gradient of the descriptor centered
on atom :math:`i` with respect to its own position.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_5.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

--------------------------------------------------------------------------------

.. html_hidden::
    :toggle: Here is the final implementation for the compute function
    :before-not-html: Here is the final implementation for the compute function

    .. literalinclude:: ../../../rascaline/src/tutorials/moments/moments.rs
        :language: rust
        :start-after: [compute]
        :end-before: [compute]
        :dedent: 4

Registering the new calculator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we are done with the code for this calculator, we need to make it
available to users. The entry point for users is the `Calculator`_ struct, which
needs to be constructed from a calculator name and hyper-parameters in JSON
format.

When the user calls ``Calculator::new("calculator_name", "{\"hyper_parameters\":
1}")``, rascaline looks for ``"calculator_name"`` in the global calculator
registry, and tries to create an instance using the hyper-parameters. In order
to make our calculator available to all users, we need to add it to this
registry, in ``rascaline/src/calculator.rs``. The registry looks like this:

.. literalinclude:: ../../../rascaline/src/calculator.rs
   :language: rust
   :start-after: [calculator-registration]
   :end-before: [calculator-registration]

``add_calculator!`` is a local macro that takes three or four arguments: the
registry itself (a ``BTreeMap``), the calculator name, the struct implementing
`CalculatorBase`_ and optionally a struct to use as parameters to create the
previous one. In our case, we want to use the three arguments version in
something like ``add_calculator!(map, "geometric_moments", GeometricMoments);``.
You'll need to make sure to bring your new calculator in scope with a `use` item.

Additionally, you may want to add a convenience class in Python for our new
calculator. For this, you can add a class like this to
``python/rascaline/calculators.py``:

.. code-block:: python

   class GeometricMoments(CalculatorBase):
    """ TODO: documentation """

      def __init__(self, cutoff, max_moment, gradients):
         parameters = {
               "cutoff": cutoff,
               "max_moment": max_moment,
               "gradients": gradients,
         }
         super().__init__("geometric_moments", parameters)


   #############################################################################

   # this allows using the calculator like this
   from rascaline import GeometricMoments
   calculator = GeometricMoments(cutoff=3.5, max_moment=6, gradients=False)

   # instead of
   from rascaline.calculators import CalculatorBase
   calculator = CalculatorBase(
      "geometric_moments",
      {"cutoff": 3.5, "max_moment": 6, "gradients": False},
   )

We have now finished our implementation of the geometric moments calculator! In
the next steps, we'll see how to write tests to ensure the calculator works and
how to write some documentation for it.

Testing the new calculator
--------------------------

Before we can release our new calculator in the world, we need to make sure it
currently behaves as intended, and that we have a way to ensure it continues to
behave as intended as the code changes. To achieve both goals, rascaline uses
unit tests and regression tests. Unit tests are written in the same file as the
main part of the code, in a ``tests`` module, and are expected to test high
level properties of the code. For example, unit tests allow to check that the
computed gradient match the derivatives of the computed values; or that the
right values are computed when the users requests a subset of samples &
features. On the other hand, regression tests check the exact values produced by
a given calculator on a specific system; and that these values stay the same as
we modify the code, for example when trying to optimize it. These regression
tests live in the ``rascaline/tests`` folder, with one file per test.

This tutorial will focus on unit tests and introduce some utilities for tests
that should apply to all calculators. To write regression tests, you should take
inspiration from existing tests such as ``spherical-expansion`` test. Each Rust
file in ``rascaline/tests`` is associated with a Python file in
``rascaline/tests/data`` used to generate the values the regression test is
checking, so you'll need one of these as well.

Testing properties
^^^^^^^^^^^^^^^^^^

If this is the first time you are writing tests in Rust, you should read the
`corresponding chapter
<https://doc.rust-lang.org/stable/book/ch11-00-testing.html>`_ in the official
Rust book for a great introduction to this subject.

Depending on the representation you are working with, you should write tests
that check the fundamental properties of this representation. For example, for
our geometric moments representation, the first moment (with order 0) should
always be the number of neighbor of the current species over the total number of
neighbors. A test checking this property would look like this:

.. literalinclude:: ../../../rascaline/src/tutorials/moments/moments.rs
   :language: rust
   :start-after: [property-test]
   :end-before: [property-test]

The ``rascaline::systems::test_utils::test_systems`` function provides a couple
of very simple systems to be used for testing.

Testing partial calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One properties that all calculators must respect is that computing only a subset
of samples or feature should give the same values as computing everything.
Rascaline provides a function (``calculators::tests_utils::compute_partial``) to
check this for you, simplifying the tests a bit. Here is how one can use it with
the ``GeometricMoments`` calculator:

.. literalinclude:: ../../../rascaline/src/tutorials/moments/moments.rs
   :language: rust
   :start-after: [partial-test]
   :end-before: [partial-test]


Testing gradients
^^^^^^^^^^^^^^^^^

If a calculator can compute gradients, it is a good idea to check if the
gradient does match the finite differences definition of derivatives. Rascaline
provides ``calculators::tests_utils::finite_difference`` to help check this.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/moments.rs
   :language: rust
   :start-after: [finite-differences-test]
   :end-before: [finite-differences-test]

Documenting the new calculator
------------------------------

.. warning:: Work in progress

    This section of the documentation is not yet written
