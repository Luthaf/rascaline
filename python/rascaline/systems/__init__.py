from .base import SystemBase
from .ase import AseSystem, HAVE_ASE
from .chemfiles import ChemfilesSystem, HAVE_CHEMFILES


def wrap_system(system):
    """
    This function is automatically called on all systems passed to
    :py:func:`rascaline.calculators.CalculatorBase.compute`. It wraps different
    systems implementation into the right class to make them compatible with
    rascaline. The following system types are supported:

    - `ase.Atoms`_: the Atomistic Simulation Environment Atoms class
    - `chemfiles.Frame`_: chemfiles' Frame type

    If ``system`` is already a subclass of :py:class:`rascaline.SystemBase`, it
    is returned as-is.

    :param system: external system of one of the above type

    :returns: a specialized instance of :py:class:`rascaline.SystemBase`

    .. _ase.Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
    .. _chemfiles.Frame: http://chemfiles.org/chemfiles.py/latest/reference/frame.html
    """
    if isinstance(system, SystemBase):
        return system

    if HAVE_ASE and AseSystem.can_wrap(system):
        return AseSystem(system)

    if HAVE_CHEMFILES and ChemfilesSystem.can_wrap(system):
        return ChemfilesSystem(system)

    raise TypeError(f"unknown system type: {type(system)}")
