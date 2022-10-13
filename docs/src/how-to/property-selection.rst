Property Selection
==================

This examples shows how to only compute a subset of properties for each sample with 
a given rascaline representation. In particular, we will use the SOAP power spectrum representation, 
and select the most significant features within a single block using farthest point sampling (FPS). 
We will run the calculation for all atoms in a structure file, the path to which should be given as the first command line argument.

This is useful if we are interested in the contribution of individual features to the result, 
or if we want to reduce the computational cost by using only part of the features for our model. 

The first part of this example repeats the :ref:`userdoc-how-to-computing-soap`, so we 
suggest that you read it initially. 

.. tabs::

    .. group-tab:: Python

        .. literalinclude:: ../../../python/examples/property-selection.py
            :language: python