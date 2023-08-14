import pytest
import numpy as np

import ase.io
import numpy as np

import rascaline
import equistore
from equistore import Labels, TensorBlock, TensorMap
import equistore.operations

from rascaline.utils import clebsch_gordan

def random_equivariant_array(n_samples=30, n_q_properties=10, n_p_properties=8, l=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    equi_l_array = np.random.rand(n_samples, 2*l+1, n_q_properties)
    return equi_l_array

def equivariant_combinable_arrays(n_samples=30, n_q_properties=10, n_p_properties=8, l1=2, l2=1, lam=2, seed=None):
    # check if valid blocks
    assert abs(l1 - l2) <= lam and lam <= l1 + l2, f"(l1={l1}, l2={l2}, lam={lam} is not valid combination, |l1-l2| <= lam <= l1+l2 must be valid"
    if seed is not None:
        np.random.seed(seed)

    equi_l1_array = random_equivariant_array(n_samples, n_q_properties, n_p_properties, l=l1)
    equi_l2_array = random_equivariant_array(n_samples, n_q_properties, n_p_properties, l=l2)
    return equi_l1_array, equi_l2_array, lam

@pytest.mark.parametrize(
    "equi_l1_array, equi_l2_array, lam",
    [equivariant_combinable_arrays(seed=51)],
)
def test_clebsch_gordan_combine_dense_sparse_agree(equi_l1_array, equi_l2_array, lam):
    cg_cache_sparse = clebsch_gordan.ClebschGordanReal(lambda_max=lam, sparse=True)
    out_sparse = clebsch_gordan._clebsch_gordan_combine_sparse(equi_l1_array, equi_l2_array, lam, cg_cache_sparse)

    cg_cache_dense = clebsch_gordan.ClebschGordanReal(lambda_max=lam, sparse=False)
    out_dense = clebsch_gordan._clebsch_gordan_combine_dense(equi_l1_array, equi_l2_array, lam, cg_cache_dense)

    assert np.allclose(out_sparse, out_dense)

@pytest.fixture
def h2o_frame():
    return ase.Atoms('H2O', positions=[[-0.526383, -0.769327, -0.029366],
                                       [-0.526383,  0.769327, -0.029366],
                                       [ 0.066334,  0.000000,  0.003701]])

def test_n_body_iteration_single_center_dense_sparse_agree(h2o_frame):
    lmax = 5
    lambdas = np.array([0, 2])
    rascal_hypers = {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 6,  # Exclusive
        "max_angular": lmax,  # Inclusive
        "atomic_gaussian_width": 0.2,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    n_body_sparse = clebsch_gordan.n_body_iteration_single_center(
        [h2o_frame],
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=True
    )

    n_body_dense = clebsch_gordan.n_body_iteration_single_center(
        [h2o_frame],
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=False
    )

    assert equistore.operations.allclose(n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8)


#def test_combine_single_center_orthogonality(h2o_frame):
#    lmax = 5
#    lambdas = np.array([0, 2])
#    rascal_hypers = {
#        "cutoff": 3.0,  # Angstrom
#        "max_radial": 6,  # Exclusive
#        "max_angular": lmax,  # Inclusive
#        "atomic_gaussian_width": 0.2,
#        "radial_basis": {"Gto": {}},
#        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
#        "center_atom_weight": 1.0,
#    }
#
#    frames = [ase.Atoms('H2O', positions=[[-0.526383, -0.769327, -0.029366],
#                                          [-0.526383,  0.769327, -0.029366],
#                                          [ 0.066334,  0.000000,  0.003701]])]
#
#    # Generate a rascaline SphericalExpansion, for only the selected samples if
#    # applicable
#    calculator = rascaline.SphericalExpansion(**rascal_hypers)
#    nu1_tensor = calculator.compute(frames, selected_samples=selected_samples)
#    combined_tensor = nu1_tensor.copy()
#
#    cg_cache = ClebschGordanReal(lambda_cut, use_sparse)
#    lambdas = np.array([0, 2])
#
#    combined_tensor = _combine_single_center(
#        tensor_1=combined_tensor,
#        tensor_2=nu1_tensor,
#        lambdas=lambdas,
#        cg_cache=cg_cache,
#        use_sparse=Ture,
#        only_keep_parity=keep_parity,
#    )
#
#    n_body_sparse = clebsch_gordan.n_body_iteration_single_center(
#        nu_target=3,
#        lambdas=lambdas,
#        lambda_cut=lmax * 2,
#        species_neighbors=[1, 8, 6],
#        use_sparse=True
#    )
#
#    np.linalg.norm
#    n_body_sparse
#


# old test that become decprecated through prototyping on API
#def test_soap_kernel():
#    """
#    Tests if we get the same result computing SOAP from spherical expansion coefficients using GC utils
#    as when using SoapPowerSpectrum.
#
#    """
#    frames = ase.Atoms('HO',
#            positions=[[0., 0., 0.], [1., 1., 1.]],
#            pbc=[False, False, False])
#
#
#    rascal_hypers = {
#        "cutoff": 3.0,  # Angstrom
#        "max_radial": 2,  # Exclusive
#        "max_angular": 3,  # Inclusive
#        "atomic_gaussian_width": 0.2,
#        "radial_basis": {"Gto": {}},
#        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
#        "center_atom_weight": 1.0,
#    }
#
#    calculator = rascaline.SphericalExpansion(**rascal_hypers)
#    nu1 = calculator.compute(frames)
#    nu1 = nu1.keys_to_properties("species_neighbor")
#
#    lmax = 1
#    lambdas = np.arange(lmax)
#
#    clebsch_gordan._create_combined_keys(nu1.keys, nu1.keys, lambdas)
#    cg_cache = clebsch_gordan.ClebschGordanReal(l_max=rascal_hypers['max_angular'])
#    soap_cg = clebsch_gordan._combine_single_center(nu1, nu1, lambdas, cg_cache)
#    soap_cg = soap_cg.keys_to_properties(["spherical_harmonics_l"]).keys_to_samples(["species_center"])
#    n_samples = len(soap_cg[0].samples)
#    kernel_cg = np.zeros((n_samples, n_samples))
#    for key, block in soap_cg.items():
#        kernel_cg += block.values.squeeze() @ block.values.squeeze().T
#
#    calculator = rascaline.SoapPowerSpectrum(**rascal_hypers)
#    nu2 = calculator.compute(frames)
#    soap_rascaline = nu2.keys_to_properties(["species_neighbor_1", "species_neighbor_2"]).keys_to_samples(["species_center"])
#    kernel_rascaline = np.zeros((n_samples, n_samples))
#    for key, block in soap_rascaline.items():
#        kernel_rascaline += block.values.squeeze() @ block.values.squeeze().T
#
#    # worries me a bit that the rtol is shit, might be missing multiplicity?
#    assert np.allclose(kernel_cg, kernel_rascaline, atol=1e-9, rtol=1e-1)
#
#def test_soap_zeros():
#    """
#    Tests if the l1!=l2 values are zero when computing the 3-body invariant (SOAP)
#    """
#    frames = ase.Atoms('HO',
#            positions=[[0., 0., 0.], [1., 1., 1.]],
#            pbc=[False, False, False])
#
#
#    rascal_hypers = {
#        "cutoff": 3.0,  # Angstrom
#        "max_radial": 2,  # Exclusive
#        "max_angular": 3,  # Inclusive
#        "atomic_gaussian_width": 0.2,
#        "radial_basis": {"Gto": {}},
#        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
#        "center_atom_weight": 1.0,
#    }
#
#    calculator = rascaline.SphericalExpansion(**rascal_hypers)
#    nu1 = calculator.compute(frames)
#    nu1 = nu1.keys_to_properties("species_neighbor")
#
#    lmax = 1
#    lambdas = np.arange(lmax)
#
#    clebsch_gordan._create_combined_keys(nu1.keys, nu1.keys, lambdas)
#    cg_cache = clebsch_gordan.ClebschGordanReal(l_max=rascal_hypers['max_angular'])
#    soap_cg = clebsch_gordan._combine_single_center(nu1, nu1, lambdas, cg_cache)
#    soap_cg.keys_to_properties("spherical_harmonics_l")
#    sliced_blocks = []
#    for key, block in soap_cg.items():
#        idx = block.properties.values[:, block.properties.names.index("l1")] != block.properties.values[:, block.properties.names.index("l2")]
#        sliced_block = equistore.slice_block(block, "properties", Labels(names=block.properties.names, values=block.properties.values[idx]))
#        sliced_blocks.append(sliced_block)
#
#        assert np.allclose(sliced_block.values, np.zeros(sliced_block.values.shape))
