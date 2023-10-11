"""
Module to compute Clebsh-Gordan coefficients and perform CG iterations. Also
contains a wrapper function for computing lambda-SOAP.

Note: this is legacy code and only used as reference.
"""

import re
from typing import Optional, Sequence, Union

import metatensor
import numpy as np
import wigners
from metatensor import Labels, TensorBlock, TensorMap

import rascaline


class ClebschGordanReal:
    """
    Class for generating CG coefficients.
    """

    def __init__(self, l_max):
        self._l_max = l_max
        self._cg = {}

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for L in range(0, self._l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(self._l_max + 1):
            for l2 in range(self._l_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[L].T

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg

    def combine(self, rho1, rho2, L):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError("Requested CG entry has not been precomputed")

        n_items = rho1.shape[0]
        n_features = rho1.shape[2]
        if rho1.shape[0] != rho2.shape[0] or rho1.shape[2] != rho2.shape[2]:
            raise IndexError("Cannot combine differently-shaped feature blocks")

        rho = np.zeros((n_items, 2 * L + 1, n_features))
        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M] += rho1[:, m1, :] * rho2[:, m2, :] * cg

        return rho

    def combine_einsum(self, rho1, rho2, L, combination_string):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = rho1.shape[0]
        if rho1.shape[0] != rho2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        features = np.einsum(combination_string, rho1[:, 0, ...], rho2[:, 0, ...]).shape
        rho = np.zeros((n_items, 2 * L + 1) + features[1:])

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M, ...] += np.einsum(
                        combination_string, rho1[:, m1, ...], rho2[:, m2, ...] * cg
                    )

        return rho

    def couple(self, decoupled, iterate=0):
        """
        Goes from an uncoupled product basis to a coupled basis. A
        (2l1+1)x(2l2+1) matrix transforming like the outer product of Y^m1_l1
        Y^m2_l2 can be rewritten as a list of coupled vectors, each transforming
        like a Y^L irrep.

        The process can be iterated: a D dimensional array that is the product
        of D Y^m_l can be turned into a set of multiple terms transforming as a
        single Y^M_L.

        decoupled: array or dict
            (...)x(2l1+1)x(2l2+1) array containing coefficients that transform
            like products of Y^l1 and Y^l2 harmonics. can also be called on a
            array of higher dimensionality, in which case the result will
            contain matrices of entries. If the further index also correspond to
            spherical harmonics, the process can be iterated, and couple() can
            be called onto its output, in which case the decoupling is applied
            to each entry.

        iterate: int
            calls couple iteratively the given number of times. equivalent to
            couple(couple(... couple(decoupled)))

        Returns:
        --------
        A dictionary tracking the nature of the coupled objects. When called one
        time, it returns a dictionary containing (l1, l2) [the coefficients of
        the parent Ylm] which in turns is a dictionary of coupled terms, in the
        form L:(...)x(2L+1)x(...) array. When called multiple times, it applies
        the coupling to each term, and keeps track of the additional l terms, so
        that e.g. when called with iterate=1 the return dictionary contains
        terms of the form (l3,l4,l1,l2) : { L: array }


        Note that this coupling scheme is different from the NICE-coupling where
        angular momenta are coupled from left to right as (((l1 l2) l3) l4)... )
        Thus results may differ when combining more than two angular channels.
        """

        coupled = {}

        # when called on a matrix, turns it into a dict form to which we can
        # apply the generic algorithm
        if not isinstance(decoupled, dict):
            l2 = (decoupled.shape[-1] - 1) // 2
            decoupled = {(): {l2: decoupled}}

        # runs over the tuple of (partly) decoupled terms
        for ltuple, lcomponents in decoupled.items():
            # each is a list of L terms
            for lc in lcomponents.keys():
                # this is the actual matrix-valued coupled term,
                # of shape (..., 2l1+1, 2l2+1), transforming as Y^m1_l1 Y^m2_l2
                dec_term = lcomponents[lc]
                l1 = (dec_term.shape[-2] - 1) // 2
                l2 = (dec_term.shape[-1] - 1) // 2

                # there is a certain redundance: the L value is also the last entry
                # in ltuple
                if lc != l2:
                    raise ValueError(
                        "Inconsistent shape for coupled angular momentum block."
                    )

                # in the new coupled term, prepend (l1,l2) to the existing label
                coupled[(l1, l2) + ltuple] = {}
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    Lterm = np.zeros(shape=dec_term.shape[:-2] + (2 * L + 1,))
                    for M in range(2 * L + 1):
                        for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                            Lterm[..., M] += dec_term[..., m1, m2] * cg
                    coupled[(l1, l2) + ltuple][L] = Lterm

        # repeat if required
        if iterate > 0:
            coupled = self.couple(coupled, iterate - 1)
        return coupled

    def decouple(self, coupled, iterate=0):
        """
        Undoes the transformation enacted by couple.
        """

        decoupled = {}
        # applies the decoupling to each entry in the dictionary
        for ltuple, lcomponents in coupled.items():
            # the initial pair in the key indicates the decoupled terms that generated
            # the L entries
            l1, l2 = ltuple[:2]

            # shape of the coupled matrix (last entry is the 2L+1 M terms)
            shape = next(iter(lcomponents.values())).shape[:-1]

            dec_term = np.zeros(
                shape
                + (
                    2 * l1 + 1,
                    2 * l2 + 1,
                )
            )
            for L in range(max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1):
                # supports missing L components, e.g. if they are zero because of symmetry
                if not L in lcomponents:
                    continue
                for M in range(2 * L + 1):
                    for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                        dec_term[..., m1, m2] += cg * lcomponents[L][..., M]
            # stores the result with a key that drops the l's we have just decoupled
            if not ltuple[2:] in decoupled:
                decoupled[ltuple[2:]] = {}
            decoupled[ltuple[2:]][l2] = dec_term

        # rinse, repeat
        if iterate > 0:
            decoupled = self.decouple(decoupled, iterate - 1)

        # if we got a fully decoupled state, just return an array
        if ltuple[2:] == ():
            decoupled = next(iter(decoupled[()].values()))
        return decoupled


# ===== helper functions for ClebschGordanReal


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    i_sqrt_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = i_sqrt_2 * 1j * (-1) ** m
            result[L + m, L + m] = -i_sqrt_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = i_sqrt_2 * (-1) ** m
            result[L - m, L + m] = i_sqrt_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)


# ======== Fxns used to perform CG iterations


def acdc_standardize_keys(descriptor):
    """
    Standardize the naming scheme of density expansion coefficient blocks
    (nu=1)
    """

    key_names = descriptor.keys.names
    if not "spherical_harmonics_l" in key_names:
        raise ValueError(
            "Descriptor missing spherical harmonics channel key `spherical_harmonics_l`"
        )
    blocks = []
    keys = []
    for key, block in descriptor.items():
        key = tuple(key)
        if not "inversion_sigma" in key_names:
            key = (1,) + key
        if not "order_nu" in key_names:
            key = (1,) + key
        keys.append(key)
        property_names = _remove_suffix(block.properties.names, "_1")
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(property_names, block.properties.values),
            )
        )

    if not "inversion_sigma" in key_names:
        key_names = ["inversion_sigma"] + key_names
    if not "order_nu" in key_names:
        key_names = ["order_nu"] + key_names

    return TensorMap(
        keys=Labels(names=key_names, values=np.asarray(keys, dtype=np.int32)),
        blocks=blocks,
    )


def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    other_keys_match=None,
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("inversion_sigma", "spherical_harmonics_l",
    "order_nu"). The automatically-determined naming of output features can be
    overridden by giving a list of "feature_names". By defaults, all other key
    labels are combined in an "outer product" mode, i.e. if there is a key-side
    neighbor_species in both x_a and x_b, the returned keys will have two
    neighbor_species labels, corresponding to the parent features. By providing
    a list `other_keys_match` of keys that should match, these are not
    outer-producted, but combined together. for instance, passing `["species
    center"]` means that the keys with the same species center will be combined
    together, but yield a single key with the same species_center in the
    results.
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    other_keys_a = tuple(
        name
        for name in x_a.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )
    other_keys_b = tuple(
        name
        for name in x_b.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )

    if other_keys_match is None:
        OTHER_KEYS = [k + "_a" for k in other_keys_a] + [k + "_b" for k in other_keys_b]
    else:
        OTHER_KEYS = (
            other_keys_match
            + [
                k + ("_a" if k in other_keys_b else "")
                for k in other_keys_a
                if k not in other_keys_match
            ]
            + [
                k + ("_b" if k in other_keys_a else "")
                for k in other_keys_b
                if k not in other_keys_match
            ]
        )

    # we assume grad components are all the same
    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).properties.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).properties.names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a.items():
        lam_a = index_a["spherical_harmonics_l"]
        sigma_a = index_a["inversion_sigma"]
        order_a = index_a["order_nu"]
        properties_a = (
            block_a.properties
        )  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples

        # and x_b
        for index_b, block_b in x_b.items():
            lam_b = index_b["spherical_harmonics_l"]
            sigma_b = index_b["inversion_sigma"]
            order_b = index_b["order_nu"]
            properties_b = block_b.properties
            samples_b = block_b.samples

            if other_keys_match is None:
                OTHERS = tuple(index_a[name] for name in other_keys_a) + tuple(
                    index_b[name] for name in other_keys_b
                )
            else:
                OTHERS = tuple(
                    index_a[k] for k in other_keys_match if index_a[k] == index_b[k]
                )
                # skip combinations without matching key
                if len(OTHERS) < len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                OTHERS = OTHERS + tuple(
                    index_a[k] for k in other_keys_a if k not in other_keys_match
                )
                OTHERS = OTHERS + tuple(
                    index_b[k] for k in other_keys_b if k not in other_keys_match
                )

            if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                # we hard-code a combination method where b can be a pair descriptor. this needs some work to be general and robust
                # note also that this assumes that structure, center are ordered in the same way in the centred and neighbor descriptors
                neighbor_slice = []
                smp_a, smp_b = 0, 0
                while smp_b < samples_b.shape[0]:
                    if samples_b[smp_b][["structure", "center"]] != samples_a[smp_a]:
                        smp_a += 1
                    neighbor_slice.append(smp_a)
                    smp_b += 1
                neighbor_slice = np.asarray(neighbor_slice)
            else:
                neighbor_slice = slice(None)

            # determines the properties that are in the select list
            sel_feats = []
            sel_idx = []
            sel_feats = (
                np.indices((len(properties_a), len(properties_b))).reshape(2, -1).T
            )

            prop_ids_a = []
            prop_ids_b = []
            for n_a, f_a in enumerate(properties_a):
                prop_ids_a.append(tuple(f_a) + (lam_a,))
            for n_b, f_b in enumerate(properties_b):
                prop_ids_b.append(tuple(f_b) + (lam_b,))
            prop_ids_a = np.asarray(prop_ids_a)
            prop_ids_b = np.asarray(prop_ids_b)
            sel_idx = np.hstack(
                [prop_ids_a[sel_feats[:, 0]], prop_ids_b[sel_feats[:, 1]]]
            )
            if len(sel_feats) == 0:
                continue
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                NU = order_a + order_b
                KEY = (
                    NU,
                    S,
                    L,
                ) + OTHERS
                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = block_b.samples
                    if grad_components is not None:
                        X_grads[KEY] = []
                        X_grad_samples[KEY] = block_b.gradient("positions").samples

                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )
                # do gradients, if they are present...
                if grad_components is not None:
                    grad_a = block_a.gradient("positions")
                    grad_b = block_b.gradient("positions")
                    grad_a_data = np.swapaxes(grad_a.data, 1, 2)
                    grad_b_data = np.swapaxes(grad_b.data, 1, 2)
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[grad_a.samples["sample"]][
                            neighbor_slice, :, sel_feats[:, 0]
                        ],
                        grad_b_data[..., sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[grad_b.samples["sample"]][:, :, sel_feats[:, 1]],
                        grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    )

                # now loop over the selected features to build the blocks

                X_idx[KEY].append(sel_idx)
                X_blocks[KEY].append(one_shot_blocks)
                if grad_components is not None:
                    X_grads[KEY].append(one_shot_grads)

    # turns data into sparse storage format (and dumps any empty block in the
    # process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[2]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
            ["spherical_harmonics_m"],
            np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1),
        )
        newblock = TensorBlock(
            # feature index must be last
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(
                feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)
            ),
        )
        if grad_components is not None:
            grad_data = np.swapaxes(np.concatenate(X_grads[KEY], axis=-1), 2, 1)
            newblock.add_gradient(
                "positions",
                data=grad_data,
                samples=X_grad_samples[KEY],
                components=[grad_components[0], sph_components],
            )
        nz_blk.append(newblock)
    X = TensorMap(
        Labels(
            ["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS,
            np.asarray(nz_idx, dtype=np.int32),
        ),
        nz_blk,
    )
    return X


def cg_increment(
    x_nu,
    x_1,
    clebsch_gordan=None,
    lcut=None,
    other_keys_match=None,
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""

    nu = x_nu.keys["order_nu"][0]

    feature_roots = _remove_suffix(x_1.block(0).properties.names)

    if nu == 1:
        feature_names = (
            [root + "_1" for root in feature_roots]
            + ["l_1"]
            + [root + "_2" for root in feature_roots]
            + ["l_2"]
        )
    else:
        feature_names = (
            [x_nu.block(0).properties.names]
            + ["k_" + str(nu + 1)]
            + [root + "_" + str(nu + 1) for root in feature_roots]
            + ["l_" + str(nu + 1)]
        )

    return cg_combine(
        x_nu,
        x_1,
        feature_names=feature_names,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
        other_keys_match=other_keys_match,
    )


def _remove_suffix(names, new_suffix=""):
    suffix = re.compile("_[0-9]?$")
    rname = []
    for name in names:
        match = suffix.search(name)
        if match is None:
            rname.append(name + new_suffix)
        else:
            rname.append(name[: match.start()] + new_suffix)
    return rname


def lambda_soap_vector(
    frames: list,
    rascal_hypers: dict,
    lambda_filter: Optional[Union[None, Sequence[int]]] = None,
    sigma_filter: Optional[Union[None, Sequence[int]]] = None,
    lambda_cut: Optional[int] = None,
    selected_samples: Optional[Labels] = None,
    neighbor_species: Optional[Sequence[int]] = None,
) -> TensorMap:
    """
    Takes a list of frames of ASE loaded frames and a dict of Rascaline
    hyperparameters and generates a lambda-SOAP (i.e. nu=2) representation.

    Passing a subset of samples in `selected_samples` can be used to, for
    instance, only calculate the features for a subset of the strutcures passed
    in `frames`. For instance: `selected_samples = Labels(names=["structure"],
    values[4, 5, 6])` will only calculate the lambda-features for structures
    indexed by 4, 5, 6.

    :param frames: a list of structures generated by the ase.io function.
    :param rascal_hypers: a dict of hyperparameters used to calculate the atom
        density correlation calculated with rascaline SphericalExpansion
    :param lambda_cut: an int of the maximum lambda value to compute
        combinations for. If none, the 'max_angular' value in `rascal_hypers`
        will be used instead.
    :param selected_samples: a Labels object that defines which samples, as a
        subset of the total samples in `frames` (i.e. atomic environments or
        structures) to perform the calculation on.
    :param neighbor_species: a list of int that correspond to the atomic charges
        of all the neighbour species that you want to be in your properties (or
        features) dimension. This list may contain charges for atoms that don't
        appear in ``frames``, but are included anyway so that the one can
        enforce consistent properties dimension size with other lambda-feature
        vectors.
    :param even_parity_only: a bool that determines whether to only include the
        key/block pairs with even parity under rotation, i.e. sigma = +1.
        Defaults to false, where both parities are included.
    :param save_dir: a str of the absolute path to the directory where the
        TensorMap of the calculated lambda-SOAP representation and pickled
        ``rascal_hypers`` dict should be written. If none, the TensorMap will not be
        saved.

    :return: a TensorMap of the lambda-SOAP feature vector for the selected
        samples of the input frames.
    """
    # Generate Rascaline spherical expansion
    if lambda_cut is None:
        lambda_cut = 2 * rascal_hypers["max_angular"]
    else:
        if lambda_cut > 2 * rascal_hypers["max_angular"]:
            raise ValueError(
                "As this function generates 2-body features (nu=2), "
                "`lambda_cut` cannot be more than 2 x rascal_hypers['max_angular'] "
                f"or less than rascal_hypers['max_angular']. Received {lambda_cut}."
            )
    # Pre-calculate ClebschGordan coefficients
    cg = ClebschGordanReal(l_max=lambda_cut)

    # Generate descriptor via Spherical Expansion
    calculator = rascaline.SphericalExpansion(**rascal_hypers)
    nu1 = calculator.compute(frames, selected_samples=selected_samples)

    # nu=1 features
    nu1 = acdc_standardize_keys(nu1)

    # Move "species_neighbor" sparse keys to properties with enforced atom
    # charges if ``neighbor_species`` is specified. This is required as the CG
    # iteration code currently does not handle neighbour species padding
    # automatically.
    keys_to_move = "species_neighbor"
    if neighbor_species is not None:
        keys_to_move = Labels(
            names=(keys_to_move,),
            values=np.array(neighbor_species).reshape(-1, 1),
        )
    nu1 = nu1.keys_to_properties(keys_to_move=keys_to_move)

    # Combined nu=1 features to generate nu=2 features. lambda-SOAP is defined
    # as just the nu=2 features.
    lsoap = cg_increment(
        nu1,
        nu1,
        clebsch_gordan=cg,
        lcut=lambda_cut,
        other_keys_match=["species_center"],
    )

    # Clean the lambda-SOAP TensorMap. Drop the order_nu key name as this is by
    # definition 2 for all keys.
    lsoap = metatensor.remove_dimension(lsoap, axis="keys", name="order_nu")

    # # Drop all odd parity keys/blocks
    # if even_parity_only:
    #     keys_to_drop = Labels(
    #         names=lsoap.keys.names,
    #         values=lsoap.keys.values[lsoap.keys.column("inversion_sigma") == -1],
    #     )
    #     lsoap = metatensor.drop_blocks(lsoap, keys=keys_to_drop)

    #     # Drop the inversion_sigma key name as this is now +1 for all blocks
    #     lsoap = metatensor.remove_dimension(lsoap, axis="keys", name="inversion_sigma")

    # Drop all blocks that don't correspond to the target lambdas
    if lambda_filter is not None:
        keys_to_drop = Labels(
            names=lsoap.keys.names,
            values=lsoap.keys.values[
                [
                    lam not in lambda_filter
                    for lam in lsoap.keys.column("spherical_harmonics_l")
                ]
            ],
        )
        lsoap = metatensor.drop_blocks(lsoap, keys=keys_to_drop)

    if sigma_filter is not None:
        if len(sigma_filter) < 2:
            keys_to_drop = Labels(
                names=lsoap.keys.names,
                values=lsoap.keys.values[
                    [
                        s not in sigma_filter
                        for s in lsoap.keys.column("inversion_sigma")
                    ]
                ],
            )
            lsoap = metatensor.drop_blocks(lsoap, keys=keys_to_drop)

            if len(np.unique(lsoap.keys.column("inversion_sigma"))) == 1:
                lsoap = metatensor.remove_dimension(
                    lsoap, axis="keys", name="inversion_sigma"
                )

    return lsoap
