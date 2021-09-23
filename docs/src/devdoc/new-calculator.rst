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
and calculators implement the `CalculatorBase`_ trait. `Descriptor`_ for their
part are implemented as a concrete struct. Traits (also called interfaces in
other languages) define contracts that the implementing code must follow, in the
form of a set of function and documented behavior for these functions.
Fulfilling this contract allow to add new systems which work with all
calculators, already implement or not; and new calculators which can use any
system, already implemented or not.

In this tutorial, our goal is to write a new struct implementing
`CalculatorBase`_. This implementation will take as input a slice of boxed
`System`_ trait objects, and using data from those fill up a `Descriptor`_.

.. _System: ../reference/rust/rascaline/systems/trait.System.html
.. _CalculatorBase: ../reference/rust/rascaline/calculators/trait.CalculatorBase.html
.. _Descriptor: ../reference/rust/rascaline/struct.Descriptor.html

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
will contain three fields: ``cutoff`` to store the cutoff radius, ``max_moment``
to store the maximal moment to compute, and ``gradients`` to indicate wether we
want to compute gradients of the descriptor or not.

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

We'll go over these functions one by one, explaining what they do as we go.
There are two groups of functions --- used to communicate metadata about the
calculator and the descriptor --- and the main function is ``compute``, which
does the main part of the work.

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
   :start-after: [CalculatorBase::get_parameters]
   :end-before: [CalculatorBase::get_parameters]
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

Descriptor metadata
^^^^^^^^^^^^^^^^^^^

The next set of functions in the `CalculatorBase`_ trait is used to communicate
metadata about the descriptor itself, and called by the concrete `Calculator`_
struct when initializing and allocating the corresponding memory.

.. _CalculatorBase: ../reference/rust/rascaline/calculators/trait.CalculatorBase.html
.. _Calculator: ../reference/rust/rascaline/struct.Calculator.html

First, ``compute_gradients`` indicates wether this calculator instance does or
not compute gradients. In our case, this is controlled by the value of
``self.gradients``.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::compute_gradients]
   :end-before: [CalculatorBase::compute_gradients]
   :dedent: 4

Then, we have three functions which work together to define what features are
computed by he current descriptor. ``features_names`` defines the name
associated with the different indexes in the features. In our case, there is
only one such index, giving the power :math:`k` used to compute the moment.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::features_names]
   :end-before: [CalculatorBase::features_names]
   :dedent: 4

The ``features`` function creates the set of `Indexes`_ used by default by this
calculator. This set of features will be used if the user does not pass a set of
selected features to `Calculator::compute`_. Here, we simply compute all moments
up to ``self.max_moment``. We build the set of indexes using an
`IndexesBuilder`_, and fill it with slices of `IndexValue`_ containing only 1
element (since we only have one index in the features).

.. _Indexes: ../reference/rust/rascaline/descriptor/struct.Indexes.html
.. _IndexValue: ../reference/rust/rascaline/descriptor/struct.IndexValue.html
.. _IndexesBuilder: ../reference/rust/rascaline/descriptor/struct.IndexesBuilder.html
.. _Calculator::compute: ../reference/rust/rascaline/struct.Calculator.html#method.compute

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::features]
   :end-before: [CalculatorBase::features]
   :dedent: 4

Finally, the ``check_features`` function will be called to verify that
user-provided features are valid, for example to request the calculation of only
a subset of values after feature selection. For the example, we check that the
value of :math:`k` is below ``self.max_moment``, although we could iin theory
accept any positive integer here.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::check_features]
   :end-before: [CalculatorBase::check_features]
   :dedent: 4

The only remaining piece of metadata required to create a `Descriptor`_ is the
definition of samples. Here, we could have used the same strategy as for
features with the three functions we just wrote. However since we expect that
multiple calculators will create the same kind of samples, we provide some
pre-defined `SamplesBuilder`_. For a two body representation which includes
species information (our case here), we can use `TwoBodiesSpeciesSamples`_. We
don't have self contribution (pair between the center and itself), so we can use
``TwoBodiesSpeciesSamples::new()`` instead of
``TwoBodiesSpeciesSamples::with_self_contribution()``.

.. _SamplesBuilder: ../reference/rust/rascaline/descriptor/trait.SamplesBuilder.html
.. _TwoBodiesSpeciesSamples: ../reference/rust/rascaline/descriptor/struct.TwoBodiesSpeciesSamples.html

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s2_metadata.rs
   :language: rust
   :start-after: [CalculatorBase::samples_builder]
   :end-before: [CalculatorBase::samples_builder]
   :dedent: 4

We are now done defining the metadata associated with our ``GeometricMoments``
calculator! In the next section, we'll go over the actual calculation of the
representation, and how to use the functions provided by `System`_.

The compute function
^^^^^^^^^^^^^^^^^^^^

We are finally approaching the most important function in `CalculatorBase`_,
``compute``. This function takes as input a list of systems and a descriptor in
which to write the results of the calculation. The function also returns a
`Result`_, to be able to indicate that an error was reached during the
calculation.

The descriptor is initialized by the concrete `Calculator`_ struct, according to
the parameters provided by the user. In particular, the descriptor will only
contain samples and features requested by th user, meaning that the code in
``compute`` should check whether a particular sample (respectively feature) is
present in ``descriptor.samples`` (resp. ``descriptor.features``) before
computing it.

This being said, let's start writing our ``compute`` function. We'll defensively
check that the descriptor samples & features match what we expect from them, and
return a unit value ``()`` wrapped in ``Ok`` at the end of the function.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_1.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

From here, the easiest way to implement our geometric moments descriptor is to
iterate over the systems, and then iterate over the pairs in the system. Before
we can get the pairs with ``system.pairs``, we need to compute the neighbors
list for our current cutoff, using ``system.compute_neighbors``, which requires
a mutable reference to the system to be able to store the list of computed pairs
(hence the iteration using ``systems.iter_mut()``).

All the functions on the `System`_ trait return `Result`_, but in contrary to
the ``get_parameters`` function above, we want to send the possible errors back
to the user so that they can deal with them as they want. The question mark
``?`` operator does exactly that: if the value returned by the called function
is ``Err(e)``, ``?`` immediately returns ``Err(e)``; and if the result is
``Ok(v)``, ``?`` extract the ``v`` and the execution continues.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_2.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

For each pair, we now have to check wether the corresponding sample was
requested by the user and is present in the descriptor. To this end, we can use
``descriptor.samples.position``, which gives us an ``Option<usize>``. This
option will be ``None`` is the sample was not found, and ``Some(position)``
where ``position`` is an unsigned integer if the sample was found. From the
documentation of `TwoBodiesSpeciesSamples`_, we know that the samples contains
four index values, corresponding to the index of the system, the index of the
central atom in this system, the species of this central atom and the species of
the neighboring atom.

One thing to keep in mind is that a given pair can participate to two different
samples. If two atoms ``i`` and ``j`` are closer than the cutoff, the list of
pairs will only contain the ``i-j`` pair, and not the ``j-i`` pair (it is a
so-called half neighbors list). That being said, we can get the list of species
with ``system.species()`` before the loop over pairs, and then construct the two
potential samples and check for their presence. If neither of the sample was
requested in the descriptor, then we can skip the calculation for this pair. We
also use ``system.pairs_containing`` to get the number of neighbors a given
center has.

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_3.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4


Now, we can iterate over the requested features, compute the moment for the
current pair distance, and accumulate it in the descriptor values:

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_4.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

The code to compute gradients is very similar, checking the positions of a given
gradient sample before writing to it. There are now four possible contributions
for a given pair: :math:`\partial \ket{\chi_i} / \partial r_j`, :math:`\partial
\ket{\chi_j} / \partial r_i`, :math:`\partial \ket{\chi_i} / \partial r_i` and
:math:`\partial \ket{\chi_j} / \partial r_j`, where :math:`\ket{\chi_i}` is the
representation around atom :math:`i`. Another way to say it is that in addition
to the gradients of the descriptor centered on :math:`i` with respect to atom
:math:`j`, we also need to account for the gradient of the descriptor centered
on atom :math:`i` with respect to its own position.

There are three samples for each contribution to the gradient (one for each
cartesian direction), but we know that they are stored one after the other in
the ``gradients`` array. We can exploit this and only look for the sample
corresponding to :math:`x` (which will be in a given ``row``), and then store
the gradients in the :math:`y` and :math:`z` directions at ``row + 1`` and ``row
+ 2``.

Since they are optional, we need to use ``as_ref``/``as_mut`` to get references
out of the ``descriptor.gradients`` and ``descriptor.gradients_samples`` fields.
These fields contains ``Option<T>`` values, and we use ``expect`` to
unconditionally extract the value since these fields should be set during the
initialization by the concrete ``Calculator``.

Putting everything together, the gradients contributions are computed using:

.. math::

    \frac{\partial}{\partial \vec{r_{j}}} \braket{\alpha k | \mathcal{A}_i} = \frac{\vec{r_{ij}}}{r_{ij}} \cdot \frac{k \ r_{ij}^{k - 1} \ \delta_{\alpha, \alpha_j}}{N_\text{neighbors}} = \vec{r_{ij}} \frac{k \ r_{ij}^{k - 2} \ \delta_{\alpha, \alpha_j}}{N_\text{neighbors}}

.. literalinclude:: ../../../rascaline/src/tutorials/moments/s3_compute_5.rs
   :language: rust
   :start-after: [compute]
   :end-before: [compute]
   :dedent: 4

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

When the user calls ``Calculator::new("calculator_name", "{\"json_parameters\":
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
`CalculatorBase`_. In our case, we want to use the three arguments version in
something like ``add_calculator!(map, "geometric_moments", GeometricMoments);``.

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
