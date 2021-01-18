from .base import SystemBase
from .ase import AseSystem


def wrap_system(external):
    """
    Make a single  external system compatible with rascaline. For now, only
    ase.Atoms is supported.

    :returns: a specialized instance of `SystemBase`
    """
    if isinstance(external, SystemBase):
        return external

    try:
        import ase

        if isinstance(external, ase.Atoms):
            return AseSystem(external)
    except ImportError:
        pass

    raise TypeError(f"unknown system type: {type(external)}")
