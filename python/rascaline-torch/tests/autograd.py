import ase
import torch

from rascaline.torch import SoapPowerSpectrum, SphericalExpansion, System


HYPERS = {
    "cutoff": 3,
    "max_radial": 2,
    "max_angular": 0,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {}},
}


def create_random_system(n_atoms, cell_size):
    torch.manual_seed(0)
    species = torch.randint(3, (n_atoms,), dtype=torch.int)

    cell = ase.cell.Cell.new([cell_size, 1.4 * cell_size, 0.8 * cell_size, 90, 80, 110])
    cell = torch.tensor(cell[:], dtype=torch.float64)

    positions = torch.rand((n_atoms, 3), dtype=torch.float64) @ cell

    return species, positions, cell


def test_spherical_expansion_positions_grad():
    species, positions, cell = create_random_system(n_atoms=75, cell_size=5.0)
    positions.requires_grad = True

    calculator = SphericalExpansion(**HYPERS)

    def compute(species, positions, cell):
        system = System(
            positions=positions,
            species=species,
            cell=cell,
        )
        descriptor = calculator(system)
        descriptor = descriptor.components_to_properties("spherical_harmonics_m")
        descriptor = descriptor.keys_to_properties("spherical_harmonics_l")

        descriptor = descriptor.keys_to_samples("species_center")
        descriptor = descriptor.keys_to_properties("species_neighbor")

        return descriptor.block(0).values

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )


def test_spherical_expansion_cell_grad():
    species, positions, cell = create_random_system(n_atoms=75, cell_size=5.0)

    original_cell = cell.clone()
    cell.requires_grad = True

    calculator = SphericalExpansion(**HYPERS)

    def compute(species, positions, cell):
        # modifying the cell for numerical gradients should also displace
        # the atoms
        fractional = positions @ torch.linalg.inv(original_cell)
        positions = fractional @ cell.detach()

        system = System(
            positions=positions,
            species=species,
            cell=cell,
        )
        descriptor = calculator(system)
        descriptor = descriptor.components_to_properties("spherical_harmonics_m")
        descriptor = descriptor.keys_to_properties("spherical_harmonics_l")

        descriptor = descriptor.keys_to_samples("species_center")
        descriptor = descriptor.keys_to_properties("species_neighbor")

        return descriptor.block(0).values

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_positions_grad():
    species, positions, cell = create_random_system(n_atoms=75, cell_size=5.0)
    positions.requires_grad = True

    calculator = SoapPowerSpectrum(**HYPERS)

    def compute(species, positions, cell):
        system = System(
            positions=positions,
            species=species,
            cell=cell,
        )
        descriptor = calculator(system)

        descriptor = descriptor.keys_to_samples("species_center")
        descriptor = descriptor.keys_to_properties(
            ["species_neighbor_1", "species_neighbor_2"]
        )

        return descriptor.block(0).values

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )


def test_power_spectrum_cell_grad():
    species, positions, cell = create_random_system(n_atoms=75, cell_size=5.0)

    original_cell = cell.clone()
    cell.requires_grad = True

    calculator = SoapPowerSpectrum(**HYPERS)

    def compute(species, positions, cell):
        # modifying the cell for numerical gradients should also displace
        # the atoms
        fractional = positions @ torch.linalg.inv(original_cell)
        positions = fractional @ cell.detach()

        system = System(
            positions=positions,
            species=species,
            cell=cell,
        )
        descriptor = calculator(system)

        descriptor = descriptor.keys_to_samples("species_center")
        descriptor = descriptor.keys_to_properties(
            ["species_neighbor_1", "species_neighbor_2"]
        )

        return descriptor.block(0).values

    assert torch.autograd.gradcheck(
        compute,
        (species, positions, cell),
        fast_mode=True,
    )
