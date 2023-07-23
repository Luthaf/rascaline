import numpy as np

import ase.io
import numpy as np

import rascaline
import equistore
from equistore import Labels, TensorBlock, TensorMap
import equistore.operations

from rascaline.utils import clebsch_gordan

def test_clebsch_gordan_combine_dense_sparse_agree():
    N_SAMPLES = 30
    N_Q_PROPERTIES = 10
    N_P_PROPERTIES = 8
    L1 = 2
    L2 = 1
    LAM = 2
    arr_1 = np.random.rand(N_SAMPLES, 2*L1+1, N_Q_PROPERTIES)
    arr_2 = np.random.rand(N_SAMPLES, 2*L2+1, N_P_PROPERTIES)

    cg_cache_sparse = clebsch_gordan.ClebschGordanReal(lambda_max=LAM, sparse=True)
    out_sparse = clebsch_gordan._clebsch_gordan_combine_sparse(arr_1, arr_2, LAM, cg_cache_sparse)

    cg_cache_dense = clebsch_gordan.ClebschGordanReal(lambda_max=LAM, sparse=False)
    out_dense = clebsch_gordan._clebsch_gordan_combine_dense(arr_1, arr_2, LAM, cg_cache_dense)

    assert np.allclose(out_sparse, out_dense)


def test_n_body_iteration_single_center_dense_sparse_agree():
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

    frames = [ase.Atoms('H2O', positions=[[-0.526383, -0.769327, -0.029366],
                                          [-0.526383,  0.769327, -0.029366],
                                          [ 0.066334,  0.000000,  0.003701]])]

    n_body_sparse = clebsch_gordan.n_body_iteration_single_center(
        frames,
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=True
    )

    n_body_dense = clebsch_gordan.n_body_iteration_single_center(
        frames,
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=False
    )

    assert equistore.operations.allclose(n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8)

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
