Sample Selection
================

This examples shows how to compute a representation only for a subset of the 
available samples. In particular, we will compute the SOAP power spectrum representation
for a specific subset of atoms, out of all the atoms in a structure file.
The path to the structure file is taken from the first command line argument.

This can be useful if we are only interested in certain structures in a large 
dataset, or if we need to determine the effect of a certain type of atoms on 
some structure properties. In the following, we will look at the tools with which 
sample selection can be done in rascaline.

The first part of this example repeats the :ref:`userdoc-how-to-computing-soap`, so we 
suggest that you read it initially. 

.. tabs::

    .. group-tab:: Python

        .. literalinclude:: ../../../python/examples/sample-selection.py
            :language: python