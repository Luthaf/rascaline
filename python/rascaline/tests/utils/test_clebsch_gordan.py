import pytest
import numpy as np

import ase.io
import numpy as np

import rascaline
import equistore
from equistore import Labels, TensorBlock, TensorMap
import equistore.operations

from rascaline.utils.clebsch_gordan import ClebschGordanReal, _clebsch_gordan_combine_dense, _clebsch_gordan_combine_sparse, n_body_iteration_single_center, _combine_single_center


def random_equivariant_array(n_samples=30, n_q_properties=10, n_p_properties=8, l=1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    equi_l_array = np.random.rand(n_samples, 2*l+1, n_q_properties)
    return equi_l_array

@pytest.mark.parametrize(
    "equi_l1_array, equi_l2_array, lam_max",
    [(random_equivariant_array(l=1, seed=51), random_equivariant_array(l=2, seed=52), 2)],
)
def test_clebsch_gordan_combine_dense_sparse_agree(equi_l1_array, equi_l2_array, lam_max):
    # TODO change to a fixture that precomputes random_equivariant_arrays up to |l1-l2| <= lam max <= l1+l2
    #      as we well as CG coeffs
    cg_cache_sparse = ClebschGordanReal(lambda_max=lam_max, sparse=True)
    out_sparse = _clebsch_gordan_combine_sparse(equi_l1_array, equi_l2_array, lam_max, cg_cache_sparse)

    cg_cache_dense = ClebschGordanReal(lambda_max=lam_max, sparse=False)
    out_dense = _clebsch_gordan_combine_dense(equi_l1_array, equi_l2_array, lam_max, cg_cache_dense)

    assert np.allclose(out_sparse, out_dense)

@pytest.mark.parametrize(
    "equi_l1_array, equi_l2_array",
    [(random_equivariant_array(l=1, seed=51), random_equivariant_array(l=1, seed=52))],
)
def test_clebsch_gordan_orthogonality(equi_l1_array, equi_l2_array):
    """
    Test orthogonality relationships

    References
    ----------

    https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Orthogonality_relations
    """
    l1 = (equi_l1_array.shape[1]-1)//2
    l2 = (equi_l2_array.shape[1]-1)//2
    lam_min = abs(l1-l2)
    lam_max = l1+l2

    cg_cache_dense = ClebschGordanReal(lambda_max=lam_max, sparse=False)
    # cache cg mats
    cg_mats = {}
    for lam in range(lam_min, lam_max+1):
        cg_mats[lam] = cg_cache_dense.coeffs[(l1, l2, lam)].reshape(-1, 2*lam+1)

    # We test lam dimension
    # \sum_{-m1 \leq l1 \leq m1, -m2 \leq l2 \leq m2} 
    #           <λμ|l1m1,l2m2> <l1m1',l2m2'|λμ'> = δ_μμ'
    for lam in range(lam_min, lam_max):
        cg_mat = cg_mats[lam]
        dot_product = cg_mat.T @ cg_mat
        diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
        diag_mask[np.diag_indices(len(dot_product))] = True
        assert np.allclose(dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask])
        assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])

    # We test l1 l2 dimension
    # \sum_{|l1-l2| \leq λ \leq l1+l2} \sum_{-μ \leq λ \leq μ}
    #            <l1m1,l2m2|λμ> <λμ|l1m1,l2m2> = δ_m1m1' δ_m2m2'
    dot_product = np.zeros((len(cg_mats[0]), len(cg_mats[0])))
    for lam in range(lam_min, lam_max+1):
        cg_mat = cg_mats[lam]
        dot_product += cg_mat @ cg_mat.T
    diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
    diag_mask[np.diag_indices(len(dot_product))] = True
    assert np.allclose(dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask])
    assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])

@pytest.fixture
def h2o_frame():
    return ase.Atoms('H2O', positions=[[-0.526383, -0.769327, -0.029366],
                                       [-0.526383,  0.769327, -0.029366],
                                       [ 0.066334,  0.000000,  0.003701]])

def test_n_body_iteration_single_center_dense_sparse_agree(h2o_frame):
    lmax = 2
    lambdas = np.array([0, 2])
    rascal_hypers = {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 2,  # Exclusive
        "max_angular": lmax,  # Inclusive
        "atomic_gaussian_width": 0.2,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    n_body_sparse = n_body_iteration_single_center(
        [h2o_frame],
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=True
    )

    n_body_dense = n_body_iteration_single_center(
        [h2o_frame],
        rascal_hypers=rascal_hypers,
        nu_target=3,
        lambdas=lambdas,
        lambda_cut=lmax * 2,
        species_neighbors=[1, 8, 6],
        use_sparse=False
    )

    assert equistore.operations.allclose(n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8)


def test_combine_single_center_orthogonality(h2o_frame):
    l = 1
    lam_min = 0
    lam_max = 2*l
    rascal_hypers = {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 6,  # Exclusive
        "max_angular": l,  # Inclusive
        "atomic_gaussian_width": 0.2,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    }

    frames = [ase.Atoms('H2O', positions=[[-0.526383, -0.769327, -0.029366],
                                          [-0.526383,  0.769327, -0.029366],
                                          [ 0.066334,  0.000000,  0.003701]])]

    # Generate a rascaline SphericalExpansion, for only the selected samples if
    # applicable
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1_tensor = calculator.compute(frames)
    # Move the "species_neighbor" key to the properties. If species_neighbors is
    # passed as a list of int, sparsity can be created in the properties for
    # these species.
    keys_to_move = "species_neighbor"
    nu1_tensor = nu1_tensor.keys_to_properties(keys_to_move=keys_to_move)
    # Add "order_nu" and "inversion_sigma" key dimensions, both with values 1
    nu1_tensor = equistore.insert_dimension(
        nu1_tensor, axis="keys", name="order_nu", values=np.array([1]), index=0
    )
    nu1_tensor = equistore.insert_dimension(
        nu1_tensor, axis="keys", name="inversion_sigma", values=np.array([1]), index=1
    )
    combined_tensor = nu1_tensor.copy()

    use_sparse = True
    cg_cache = ClebschGordanReal(lam_max, sparse=use_sparse)
    lambdas = np.array(range(lam_max))

    combined_tensor = _combine_single_center(
        tensor_1=combined_tensor,
        tensor_2=nu1_tensor,
        lambdas=lambdas,
        cg_cache=cg_cache,
        use_sparse=use_sparse, # TODO remove use_sparse parameter, can be referred from cg_cache
        only_keep_parity=True,
    )
    nu_target = 1

    #order_nu  inversion_sigma  spherical_harmonics_l  species_center
    #"spherical_harmonics_l",
    combined_tensor = combined_tensor.keys_to_properties(["l1", "l2", "inversion_sigma", "order_nu"])
    combined_tensor = combined_tensor.keys_to_samples(["species_center"])
    n_samples = combined_tensor[0].values.shape[0]
    combined_tensor_values = np.hstack(
                            [combined_tensor.block(Labels("spherical_harmonics_l", np.array([[l]]))).values.reshape(n_samples, -1)
                                for l in combined_tensor.keys["spherical_harmonics_l"]])
    combined_tensor_norm = np.linalg.norm(combined_tensor_values, axis=1)

    nu1_tensor = nu1_tensor.keys_to_properties(["inversion_sigma", "order_nu"])
    nu1_tensor = nu1_tensor.keys_to_samples(["species_center"])
    nu1_tensor_values = np.hstack(
                            [nu1_tensor.block(Labels("spherical_harmonics_l", np.array([[l]]))).values.reshape(n_samples, -1)
                                for l in nu1_tensor.keys["spherical_harmonics_l"]])
    nu1_tensor_norm = np.linalg.norm(nu1_tensor_values, axis=1)
    assert np.allclose(combined_tensor_norm, nu1_tensor_norm)



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
