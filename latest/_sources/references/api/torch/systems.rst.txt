System
======

Instead of a custom ``System`` class, ``rascaline-torch`` uses the class defined
by metatensor's atomistic models facilities:
:py:class:`metatensor.torch.atomistic.System`. Rascaline provides converters
from all the supported system providers (i.e. everything in
:py:class:`rascaline.IntoSystem`) to the TorchScript compatible ``System``.

.. autofunction:: rascaline.torch.systems_to_torch
