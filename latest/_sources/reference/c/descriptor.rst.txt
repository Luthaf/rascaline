Dealing with descriptors
========================

.. doxygentypedef:: rascal_descriptor_t

The following functions operate on :c:type:`rascal_descriptor_t`:

- :c:func:`rascal_descriptor`: create new descriptors
- :c:func:`rascal_descriptor_free`: free allocated descriptors
- :c:func:`rascal_calculator_compute`: run the actual calculation
- :c:func:`rascal_descriptor_values`: get the values out of the descriptor
- :c:func:`rascal_descriptor_gradients`: get the gradients out of the descriptor
- :c:func:`rascal_descriptor_indexes`: get the values of one of the indexes of the descriptor
- :c:func:`rascal_descriptor_indexes_names`: get the names associated with one of the indexes of the descriptor
- :c:func:`rascal_descriptor_densify`: move some indexes variables from samples to features


.. doxygenfunction:: rascal_descriptor

.. doxygenfunction:: rascal_descriptor_free

.. doxygenfunction:: rascal_descriptor_values

.. doxygenfunction:: rascal_descriptor_gradients

.. doxygenfunction:: rascal_descriptor_indexes

.. doxygenfunction:: rascal_descriptor_indexes_names

.. doxygenfunction:: rascal_descriptor_densify


.. doxygenenum:: rascal_indexes
