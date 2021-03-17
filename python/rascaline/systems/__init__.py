from .base import SystemBase
from .ase import AseSystem


def wrap_system(system):
    """
    This function is automatically called on all systems passed to
    :py:func:`rascaline.calculator.CalculatorBase.compute`. It wraps different
    systems implementation into the right class to make them compatible with
    rascaline. The following system types are supported:

    - `ase.Atoms`_: the Atomistic Simulation Environment Atoms class

    :param system: external system of one of the above type

    :returns: a specialized instance of :py:class:`rascaline.SystemBase`

    .. _ase.Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    """
    if isinstance(system, SystemBase):
        return system

    try:
        import ase

        if isinstance(system, ase.Atoms):
            return AseSystem(system)
    except ImportError:
        pass

    raise TypeError(f"unknown system type: {type(system)}")
