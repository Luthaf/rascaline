System
======

Instead of a custom ``System`` class, ``featomic-torch`` uses the class defined
by metatensor's atomistic models facilities:
:py:class:`metatensor.torch.atomistic.System`. Featomic provides converters
from all the supported system providers (i.e. everything in
:py:class:`featomic.IntoSystem`) to the TorchScript compatible ``System``.

.. autofunction:: featomic.torch.systems_to_torch
