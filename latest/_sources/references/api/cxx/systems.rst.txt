Defining systems
================

There are two ways you can define systems to pass to a
:cpp:class:`rascaline::Calculator`. The easy way is to use
:cpp:class:`rascaline::BasicSystems` to read all systems defined in a file, and
run the calculation on all these systems. The more complex but also more
flexible way is to define a new child class of :cpp:class:`rascaline::System`
implementing all required functions; and then passing a vector of pointers to
the child class instances to your :cpp:class:`rascaline::Calculator`.

.. doxygenclass:: rascaline::System
    :members:

.. doxygenclass:: rascaline::BasicSystems
    :members:
