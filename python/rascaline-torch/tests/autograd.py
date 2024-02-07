import warnings

import ase
import pytest
import torch
from metatensor.torch.atomistic import System

import rascaline.torch
from rascaline.torch import SoapPowerSpectrum, SphericalExpansion


HYPERS = {
    "cutoff": 3,
    "max_radial": 10,
    "max_angular": 5,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


def _create_random_system(n_atoms, cell_size):
    torch.manual_seed(0)
    species = torch.randint(3, (n_atoms,), dtype=torch.int)

    cell = ase.cell.Cell.new([cell_size, 1.4 * cell_size, 0.8 * cell_size, 90, 80, 110])
    cell = torch.tensor(cell[:], dtype=torch.float64)

    positions = torch.rand((n_atoms, 3), dtype=torch.float64) @ cell

    return species, positions, cell


def _compute_spherical_expansion(species, positions, cell):
    system = System(
        positions=positions,
        species=species,
        cell=cell,
    )

    calculator = SphericalExpansion(**HYPERS)
    descriptor = calculator(system)
    descriptor = descriptor.components_to_properties("spherical_harmonics_m")
    descriptor = descriptor.keys_to_properties("spherical_harmonics_l")

    descriptor = descriptor.keys_to_samples("species_center")
    descriptor = descriptor.keys_to_properties("species_neighbor")

    return descriptor.block(0).values


def _compute_power_spectrum(species, positions, cell):
    system = System(
        positions=positions,
        species=species,
        cell=cell,
    )

    calculator = SoapPowerSpectrum(**HYPERS)
    descriptor = calculator(system)
    descriptor = descriptor.keys_to_samples("species_center")
    descriptor = descriptor.keys_to_properties(
        ["species_neighbor_1", "species_neighbor_2"]
    )

    return descriptor.block(0).values


def test_spherical_expansion_positions_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)
    positions.requires_grad = True

    assert torch.autograd.gradcheck(
        _compute_spherical_expansion,
        (species, positions, cell),
        fast_mode=True,
    )


def test_spherical_expansion_cell_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)

    original_cell = cell.clone()
    cell.requires_grad = True

    def compute(species, positions, cell):
        # modifying the cell for numerical gradients should also displace
        # the atoms
        fractional = positions @ torch.linalg.inv(original_cell)
        positions = fractional @ cell.detach()

        return _compute_spherical_expansion(species, positions, cell)

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_positions_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)
    positions.requires_grad = True

    assert torch.autograd.gradcheck(
        _compute_power_spectrum,
        (species, positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_positions_grad_register_autograd():
    # check autograd when registering the graph after pre-computing a representation
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)

    calculator = SoapPowerSpectrum(**HYPERS)
    precomputed = calculator(System(species, positions, cell), gradients=["positions"])

    # no grad_fn for now
    assert precomputed.block(0).values.grad_fn is None

    def compute(positions, cell):
        system = System(
            positions=positions,
            species=species,
            cell=cell,
        )

        descriptor = rascaline.torch.register_autograd(system, precomputed)
        descriptor = descriptor.keys_to_samples("species_center")
        descriptor = descriptor.keys_to_properties(
            ["species_neighbor_1", "species_neighbor_2"]
        )

        # a grad_fn have been added!
        assert descriptor.block(0).values.grad_fn is not None

        return descriptor.block(0).values

    positions.requires_grad = True
    assert torch.autograd.gradcheck(
        compute,
        (positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_cell_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)

    original_cell = cell.clone()
    cell.requires_grad = True

    def compute(species, positions, cell):
        # modifying the cell for numerical gradients should also displace
        # the atoms
        fractional = positions @ torch.linalg.inv(original_cell)
        positions = fractional @ cell.detach()
        return _compute_power_spectrum(species, positions, cell)

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_positions_grad_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)
    positions.requires_grad = True

    X = _compute_power_spectrum(species, positions, cell)
    weights = torch.rand((X.shape[-1], 1), requires_grad=True, dtype=torch.float64)

    def compute(weights):
        X = _compute_power_spectrum(species, positions, cell)
        A = X @ weights

        return torch.autograd.grad(
            outputs=A,
            inputs=positions,
            grad_outputs=torch.ones_like(A),
            retain_graph=True,
            create_graph=True,
        )[0]

    message = (
        "second derivatives with respect to positions are not implemented and "
        "will not be accumulated during backward\\(\\) calls"
    )
    computed = torch.sum(compute(weights))
    with pytest.warns(UserWarning, match=message):
        computed.backward(retain_graph=True)

    # check that double backward still allows for gradients of weights w.r.t. forces
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=message)

        assert torch.autograd.gradcheck(
            compute,
            (weights),
            fast_mode=True,
        )


def test_power_spectrum_cell_grad_grad():
    species, positions, cell = _create_random_system(n_atoms=75, cell_size=5.0)
    cell.requires_grad = True

    X = _compute_power_spectrum(species, positions, cell)
    weights = torch.rand((X.shape[-1], 1), requires_grad=True, dtype=torch.float64)

    def compute(weights):
        X = _compute_power_spectrum(species, positions, cell)
        A = X @ weights

        return torch.autograd.grad(
            outputs=A,
            inputs=cell,
            grad_outputs=torch.ones_like(A),
            retain_graph=True,
            create_graph=True,
        )[0]

    message = (
        "second derivatives with respect to cell matrix are not implemented and "
        "will not be accumulated during backward\\(\\) calls"
    )
    computed = torch.sum(compute(weights))
    with pytest.warns(UserWarning, match=message):
        computed.backward(retain_graph=True)

    # check that double backward still allows for gradients of weights w.r.t. virial
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=message)

        assert torch.autograd.gradcheck(
            compute,
            (weights),
            fast_mode=True,
        )


def test_different_device_dtype():
    # check autograd if the data is on different devices/dtypes as well
    options = [
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
    ]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        options.append((torch.device("mps:0"), torch.float32))

    if torch.cuda.is_available():
        options.append((torch.device("cuda:0"), torch.float64))

    for device, dtype in options:
        species, positions, cell = _create_random_system(n_atoms=10, cell_size=3.0)
        positions = positions.to(dtype=dtype, device=device, copy=True)
        positions.requires_grad = True
        assert positions.grad is None

        cell = cell.to(dtype=dtype, device=device, copy=True)
        cell.requires_grad = True
        assert cell.grad is None

        species = species.to(device=device, copy=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            X = _compute_power_spectrum(species, positions, cell)

        assert X.dtype == dtype
        assert X.device == device

        weights = torch.rand(
            (X.shape[-1], 1), requires_grad=True, dtype=X.dtype, device=X.device
        )
        assert weights.grad is None

        A = (X @ weights).sum()
        positions_grad = torch.autograd.grad(
            outputs=A,
            inputs=positions,
            grad_outputs=torch.ones_like(A),
            retain_graph=True,
            create_graph=True,
        )[0]

        cell_grad = torch.autograd.grad(
            outputs=A,
            inputs=cell,
            grad_outputs=torch.ones_like(A),
            retain_graph=True,
            create_graph=True,
        )[0]

        assert positions_grad.dtype == dtype
        assert positions_grad.device == device

        assert cell_grad.dtype == dtype
        assert cell_grad.device == device

        # should not error
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            positions_grad.sum().backward()
            cell_grad.sum().backward()

        assert weights.grad.dtype == dtype
        assert weights.grad.device == device
