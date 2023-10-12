import ase.io
import metatensor
import metatensor.operations
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import rascaline
from rascaline.utils.clebsch_gordan import (
    ClebschGordanReal,
    combine_single_center_to_body_order,
)
from rascaline.utils.clebsch_gordan.clebsch_gordan import (
    _combine_arrays_dense,
    _combine_arrays_sparse,
    _combine_single_center_blocks,
    combine_single_center_one_iteration,
)


def random_equivariant_array(
    n_samples=10, n_q_properties=4, n_p_properties=8, l=1, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    equi_l_array = np.random.rand(n_samples, 2 * l + 1, n_q_properties)
    return equi_l_array


class TestClebschGordan:
    lam_max = 3
    cg_cache_sparse = ClebschGordanReal(lambda_max=lam_max, sparse=True)
    cg_cache_dense = ClebschGordanReal(lambda_max=lam_max, sparse=False)

    def test_clebsch_gordan_combine_dense_sparse_agree(self):
        for l1, l2, lam in self.cg_cache_dense.coeffs.keys():
            equi_l1_array = random_equivariant_array(l=l1, seed=51)
            equi_l2_array = random_equivariant_array(l=l2, seed=53)
            out_sparse = _combine_arrays_sparse(
                equi_l1_array, equi_l2_array, lam, self.cg_cache_sparse
            )
            out_dense = _combine_arrays_dense(
                equi_l1_array, equi_l2_array, lam, self.cg_cache_dense
            )
            assert np.allclose(out_sparse, out_dense)

    @pytest.mark.parametrize("l1, l2", [(1, 2)])
    def test_clebsch_gordan_orthogonality(self, l1, l2):
        """
        Test orthogonality relationships

        References
        ----------

        https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Orthogonality_relations
        """
        assert (
            self.lam_max >= l1 + l2
        ), "Did not precompute CG coeff with high enough lambda_max"
        lam_min = abs(l1 - l2)
        lam_max = l1 + l2

        # We test lam dimension
        # \sum_{-m1 \leq l1 \leq m1, -m2 \leq l2 \leq m2}
        #           <λμ|l1m1,l2m2> <l1m1',l2m2'|λμ'> = δ_μμ'
        for lam in range(lam_min, lam_max):
            cg_mat = self.cg_cache_dense.coeffs[(l1, l2, lam)].reshape(-1, 2 * lam + 1)
            dot_product = cg_mat.T @ cg_mat
            diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
            diag_mask[np.diag_indices(len(dot_product))] = True
            assert np.allclose(
                dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask]
            )
            assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])

        # We test l1 l2 dimension
        # \sum_{|l1-l2| \leq λ \leq l1+l2} \sum_{-μ \leq λ \leq μ}
        #            <l1m1,l2m2|λμ> <λμ|l1m1,l2m2> = δ_m1m1' δ_m2m2'
        l1l2_dim = (2 * l1 + 1) * (2 * l2 + 1)
        dot_product = np.zeros((l1l2_dim, l1l2_dim))
        for lam in range(lam_min, lam_max + 1):
            cg_mat = self.cg_cache_dense.coeffs[(l1, l2, lam)].reshape(-1, 2 * lam + 1)
            dot_product += cg_mat @ cg_mat.T
        diag_mask = np.zeros(dot_product.shape, dtype=np.bool_)
        diag_mask[np.diag_indices(len(dot_product))] = True
        assert np.allclose(
            dot_product[~diag_mask], np.zeros(dot_product.shape)[~diag_mask]
        )
        assert np.allclose(dot_product[diag_mask], dot_product[diag_mask][0])

    h2o_frame = ase.Atoms(
        "H2O",
        positions=[
            [-0.526383, -0.769327, -0.029366],
            [-0.526383, 0.769327, -0.029366],
            [0.066334, 0.000000, 0.003701],
        ],
    )

    @pytest.mark.parametrize("lam_max", [2])
    def test_combine_single_center_to_body_order_dense_sparse_agree(self, lam_max):
        """
        tests if combine_single_center_to_body_order agrees for dense and sparse cg
        coeffs
        """
        assert (
            self.lam_max >= lam_max
        ), "Did not precompute CG coeff with high enough lambda_max"
        lambdas = [0, 2]
        rascal_hypers = {
            "cutoff": 3.0,  # Angstrom
            "max_radial": 2,  # Exclusive
            "max_angular": lam_max,  # Inclusive
            "atomic_gaussian_width": 0.2,
            "radial_basis": {"Gto": {}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            "center_atom_weight": 1.0,
        }
        calculator = rascaline.SphericalExpansion(**rascal_hypers)
        nu1_tensor = calculator.compute([self.h2o_frame])
        n_body_sparse = combine_single_center_to_body_order(
            nu1_tensor,
            3,  # target_body_order
            angular_cutoff=lam_max * 2,
            angular_selection=lambdas,
            parity_selection=None,
            use_sparse=True,
        )

        n_body_dense = combine_single_center_to_body_order(
            nu1_tensor,
            3,  # target_body_order
            angular_cutoff=lam_max * 2,
            angular_selection=lambdas,
            parity_selection=None,
            use_sparse=False,
        )

        assert metatensor.operations.allclose(
            n_body_sparse, n_body_dense, atol=1e-8, rtol=1e-8
        )

    @pytest.mark.parametrize("l", [1])
    def test_combine_single_center_orthogonality(self, l):
        lam_min = 0
        lam_max = 2 * l
        rascal_hypers = {
            "cutoff": 3.0,  # Angstrom
            "max_radial": 6,  # Exclusive
            "max_angular": l,  # Inclusive
            "atomic_gaussian_width": 0.2,
            "radial_basis": {"Gto": {}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
            "center_atom_weight": 1.0,
        }

        calculator = rascaline.SphericalExpansion(**rascal_hypers)
        nu1_tensor = calculator.compute([self.h2o_frame])

        # compute norm of the body order 2 tensor
        nu2_tensor = combine_single_center_to_body_order(
            nu1_tensor,
            2,  # target_body_order
            angular_cutoff=None,
            angular_selection=None,
            parity_selection=None,
            use_sparse=True,
        )
        nu2_tensor = nu2_tensor.keys_to_properties(["inversion_sigma", "order_nu"])
        nu2_tensor = nu2_tensor.keys_to_samples(["species_center"])
        n_samples = nu2_tensor[0].values.shape[0]
        nu2_tensor_values = np.hstack(
            [
                nu2_tensor.block(
                    Labels("spherical_harmonics_l", np.array([[l]]))
                ).values.reshape(n_samples, -1)
                for l in nu2_tensor.keys["spherical_harmonics_l"]
            ]
        )
        nu2_tensor_norm = np.linalg.norm(nu2_tensor_values, axis=1)

        #  compute norm of the body order 1 tensor
        nu1_tensor = nu1_tensor.keys_to_properties(["species_neighbor"])
        nu1_tensor = nu1_tensor.keys_to_samples(["species_center"])
        nu1_tensor_values = np.hstack(
            [
                nu1_tensor.block(
                    Labels("spherical_harmonics_l", np.array([[l]]))
                ).values.reshape(n_samples, -1)
                for l in nu1_tensor.keys["spherical_harmonics_l"]
            ]
        )
        nu1_tensor_norm = np.linalg.norm(nu1_tensor_values, axis=1)

        # check if the norm is equal
        assert np.allclose(nu2_tensor_norm, nu1_tensor_norm)

    # old test that become decprecated through prototyping on API
    # def test_soap_kernel():
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
    #    soap_cg = clebsch_gordan._combine_single_center(nu1, nu1, lambdas, cg_cache) # ->  needs
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
    # def test_soap_zeros():
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
    #        sliced_block = metatensor.slice_block(block, "properties", Labels(names=block.properties.names, values=block.properties.values[idx]))
    #        sliced_blocks.append(sliced_block)
    #
    #        assert np.allclose(sliced_block.values, np.zeros(sliced_block.values.shape))
