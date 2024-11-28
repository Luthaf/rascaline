.. _torch-api-reference:

TorchScript API reference
=========================

.. py:currentmodule:: featomic.torch

We provide a PyTorch C++ extension to make featomic compatible with Torch and
TorchScript in three ways:

- registering featomic calculators as special nodes in Torch's computational
  graph, allowing to use backward propagation of derivatives to compute
  gradients of arbitrary quantities with respect to atomic positions and cell
  (e.g. forces and stress when the quantity is the energy of a system);
- saving and loading calculators inside a torch Model (the calculators are
  exposed as special case of ``torch.nn.Module``)
- exporting a model trained in Python and loading it back without needing the
  Python interpreter, for example inside a pure C++ or Fortran molecular
  simulation engine.

Please refer to the :ref:`installation instructions <install-torch-script>` to
know how to install the Python and C++ sides of this library. The core classes
of featomic are documented below for an usage from Python:

.. toctree::
    :maxdepth: 1

    systems
    calculators
    clebsch-gordan

--------------------------------------------------------------------------------

If you want to use featomic's TorchScript API from C++, you might be interested
in the following documentation:

.. toctree::
    :maxdepth: 1

    cxx/index
