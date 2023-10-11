# def _apply_body_order_corrections(tensor: TensorMap) -> TensorMap:
#     """
#     Applies the appropriate prefactors to the block values of the output
#     TensorMap (i.e. post-CG combination) according to its body order.
#     """
#     return tensor


# def _normalize_blocks(tensor: TensorMap) -> TensorMap:
#     """
#     Applies corrections to the block values based on their 'leaf' l-values, such
#     that the norm is preserved.
#     """
#     return tensor


# ===== ROUGH WORK:

# def _parse_parity_selection(
#     target_body_order: int,
#     parity_selection: Union[None, int, List[int], List[List[int]]],
# ) -> List[List[int]]:
#     """
#     Returns parity filters for each CG combination step of a nu=1 tensor with
#     itself up to the target body order.

#     If a filter isn't specified by the user with `parity_selection=None`, then
#     no filter is applied, i.e. [-1, +1] is used at every iteration.

#     If a single sequence of int is specified, then this is used for the last
#     iteration only, and [-1, +1] is used for all intermediate iterations. For
#     example, if `target_body_order=4` and `parity_selection=[+1]`, then the
#     filter [[-1, +1], [-1, +1], [+1]] is returned.

#     If a sequence of sequences of int is specified, then this is assumed to be
#     the desired filter for each iteration.

#     Note: very basic checks on the validity of the parity selections are
#     performed, but these are not complete as they do not account for the angular
#     channels of the blocks. These interactions are checked downstream in
#     :py:func:`_create_combined_keys`.
#     """
#     if target_body_order < 2:
#         raise ValueError("`target_body_order` must be > 1")

#     # No filter specified: use [-1, +1] for all iterations
#     if parity_selection is None:
#         parity_selection = [[-1, +1] for _ in range(target_body_order - 1)]

#     # Parse user-defined filter: assume passed as List[int] or
#     # List[List[int]]
#     else:
#         if isinstance(parity_selection, int):
#             parity_selection = [parity_selection]
#         if not isinstance(parity_selection, List):
#             raise TypeError(
#                 "`parity_selection` must be an int, List[int], or List[List[int]]"
#             )
#         # Single filter: apply on last iteration only, use both sigmas for
#         # intermediate iterations
#         if np.all([isinstance(sigma, int) for sigma in parity_selection]):
#             parity_selection = [[-1, +1] for _ in range(target_body_order - 2)] + [parity_selection]

#     # Check parity_selection
#     assert isinstance(parity_selection, List)
#     assert len(parity_selection) == target_body_order - 1
#     assert np.all([isinstance(filt, List) for filt in parity_selection])
#     assert np.all([np.all([s in [-1, +1] for s in filt]) for filt in parity_selection])

#     return parity_selection


# def _parse_angular_selection(
#     angular_channels_1: List[int],
#     angular_channels_2: List[int],
#     target_body_order: int,
#     rascal_max_l: int,
#     angular_selection: Union[None, int, List[int], List[List[int]]],
#     lambda_cut: Union[None, int],
# ) -> List[List[int]]:
#     """
#     Parses the user-defined angular selection filters, returning 

#     If a filter isn't specified by the user with `angular_selection=None`, then
#     no filter is applied. In this case all possible lambda channels are retained
#     at each iteration. For example, if `target_body_order=4`, `rascal_max_l=5`,
#     and `lambda_cut=None`, then the returned filter is [[0, ..., 10], [0, ...,
#     15], [0, ..., 20]]. If `target_body_order=4`, `rascal_max_l=5`, and
#     `lambda_cut=10`, then the returned filter is [[0, ..., 10], [0, ..., 10],
#     [0, ..., 10]].

#     If `angular_selection` is passed a single sequence of int, then this is used
#     for the last iteration only, and all possible combinations of lambda are
#     used in intermediate iterations. For instance, if `angular_selection=[0, 1,
#     2]`, the returned filters for the 2 examples above, respectively, would be
#     [[0, ..., 10], [0, ..., 15], [0, 1, 2]] and [[0, ..., 10], [0, ..., 10], [0,
#     1, 2]].

#     If a sequence of sequences of int is specified, then this is assumed to be
#     the desired filter for each iteration and is only checked for validity
#     without modification.

#     Note: basic checks on the validity of the angular selections are performed,
#     but these are not complete as they do not account for the parity of the
#     blocks. These interactions are checked downstream in
#     :py:func:`_create_combined_keys`.
#     """
#     if target_body_order < 2:
#         raise ValueError("`target_body_order` must be > 1")

#     # Check value of lambda_cut
#     if lambda_cut is not None:
#         if not (rascal_max_l <= lambda_cut <= target_body_order * rascal_max_l):
#             raise ValueError(
#                 "`lambda_cut` must be >= `rascal_hypers['max_angular']` and <= `target_body_order`"
#                 " * `rascal_hypers['max_angular']`"
#             )

#     # No filter specified: retain all possible lambda channels for every
#     # iteration, up to lambda_cut (if specified)
#     if angular_selection is None:
#         if lambda_cut is None:
#             # Use the full range of possible lambda channels for each iteration.
#             # This is dependent on the itermediate body order.
#             angular_selection = [
#                 [lam for lam in range(0, (nu * rascal_max_l) + 1)]
#                 for nu in range(2, target_body_order + 1)
#             ]
#         else:
#             # Use the full range of possible lambda channels for each iteration,
#             # but only up to lambda_cut, independent of the intermediate body
#             # order.
#             angular_selection = [
#                 [lam for lam in range(0, lambda_cut + 1)]
#                 for nu in range(2, target_body_order + 1)
#             ]

#     # Parse user-defined filter: assume passed as List[int] or
#     # List[List[int]]
#     else:
#         if isinstance(angular_selection, int):
#             angular_selection = [angular_selection]
#         if not isinstance(angular_selection, List):
#             raise TypeError(
#                 "`angular_selection` must be an int, List[int], or List[List[int]]"
#             )
#         # Single filter: apply on last iteration only, use all possible lambdas for
#         # intermediate iterations (up to lambda_cut, if specified)
#         if np.all([isinstance(filt, int) for filt in angular_selection]):
#             if lambda_cut is None:
#                 # Use the full range of possible lambda channels for each iteration.
#                 # This is dependent on the itermediate body order.
#                 angular_selection = [
#                     [lam for lam in range(0, (nu * rascal_max_l) + 1)]
#                     for nu in range(2, target_body_order)
#                 ] + [angular_selection]

#             else:
#                 # Use the full range of possible lambda channels for each iteration,
#                 # but only up to lambda_cut, independent of the intermediate body
#                 # order.
#                 angular_selection = [
#                     [lam for lam in range(0, lambda_cut + 1)]
#                     for nu in range(2, target_body_order)
#                 ] + [angular_selection]

#         else:
#             # Assume filter explicitly defined for each iteration (checked below)
#             pass

#     # Check angular_selection
#     if not isinstance(angular_selection, List):
#         raise TypeError(
#             "`angular_selection` must be an int, List[int], or List[List[int]]"
#         )
#     if len(angular_selection) != target_body_order - 1:
#         raise ValueError(
#             "`angular_selection` must have length `target_body_order` - 1, i.e. the number of CG"
#             " iterations required to reach `target_body_order`"
#         )
#     if not np.all([isinstance(filt, List) for filt in angular_selection]):
#         raise TypeError(
#             "`angular_selection` must be an int, List[int], or List[List[int]]"
#         )
#     # Check the lambda values are within the possible range, based on each
#     # intermediate body order
#     if not np.all(
#         [
#             np.all([0 <= lam <= nu * rascal_max_l for lam in filt])
#             for nu, filt in enumerate(angular_selection, start=2)
#         ]
#     ):
#         raise ValueError(
#             "All lambda values in `angular_selection` must be >= 0 and <= `nu` *"
#             " `rascal_hypers['max_angular']`, where `nu` is the body"
#             " order created in the intermediate CG combination step"
#         )
#     # Now check that at each iteration the lambda values can actually be created
#     # from combination at the previous iteration
#     for filt_i, filt in enumerate(angular_selection):
#         if filt_i == 0:
#             # Assume that the original nu=1 tensors to be combined have all l up
#             # to and including `rascal_max_l`
#             allowed_lams = np.arange(0, (2 * rascal_max_l) + 1)
#         else:
#             allowed_lams = []
#             for l1, l2 in itertools.product(angular_selection[filt_i - 1], repeat=2):
#                 for lam in range(abs(l1 - l2), abs(l1 + l2) + 1):
#                     allowed_lams.append(lam)

#             allowed_lams = np.unique(allowed_lams)

#         if not np.all([lam in allowed_lams for lam in filt]):
#             raise ValueError(
#                 f"invalid lambda values in `angular_selection` for iteration {filt_i + 1}."
#                 f" {filt} cannot be created by combination of previous lambda values"
#                 f" {angular_selection[filt_i - 1]}"
#             )

#     return angular_selection


# def _combine_single_center(
#     tensor_1: TensorMap,
#     tensor_2: TensorMap,
#     lambdas: List[int],
#     sigmas: List[int],
#     cg_cache,
# ) -> TensorMap:
#     """
#     For 2 TensorMaps, ``tensor_1`` and ``tensor_2``, with body orders nu and 1
#     respectively, combines their blocks to form a new TensorMap with body order
#     (nu + 1).

#     Returns blocks only indexed by keys .

#     Assumes the metadata of the two TensorMaps are standardized as follows.

#     The keys of `tensor_1`  must follow the key name convention:

#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
#     "l1", "l2", ..., f"l{`nu`}", "k2", ..., f"k{`nu`-1}"]. The "lx" columns
#     track the l values of the nu=1 blocks that were previously combined. The
#     "kx" columns tracks the intermediate lambda values of nu > 1 blocks that
#     haev been combined.

#     For instance, a TensorMap of body order nu=4 will have key names
#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center",
#     "l1", "l2", "l3", "l4", "k2", "k3"]. Two nu=1 TensorMaps with blocks of
#     order "l1" and "l2" were combined to form a nu=2 TensorMap with blocks of
#     order "k2". This was combined with a nu=1 TensorMap with blocks of order
#     "l3" to form a nu=3 TensorMap with blocks of order "k3". Finally, this was
#     combined with a nu=1 TensorMap with blocks of order "l4" to form a nu=4.

#     .. math ::

#         \bra{ n_1 l_1 ; n_2 l_2 k_2 ; ... ; n{\nu-1} l_{\nu-1} k_{\nu-1} ;
#         n{\nu} l_{\nu} k_{\nu}; \lambda } \ket{ \rho^{\otimes \nu}; \lambda M }

#     The keys of `tensor_2` must follow the key name convention:

#     ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"]

#     Samples of pairs of blocks corresponding to the same chemical species are
#     equivalent in the two TensorMaps. Samples names are ["structure", "center"]

#     Components names are [["spherical_harmonics_m"],] for each block.

#     Property names are ["n1", "n2", ..., "species_neighbor_1",
#     "species_neighbor_2", ...] for each block.
#     """

#     # Get the correct keys for the combined output TensorMap
#     (
#         nux_keys,
#         keys_1_entries,
#         keys_2_entries,
#         multiplicity_list,
#     ) = _create_combined_keys(tensor_1.keys, tensor_2.keys, lambdas, sigmas)

#     # Iterate over pairs of blocks and combine
#     nux_blocks = []
#     for nux_key, key_1, key_2, multi in zip(
#         nux_keys, keys_1_entries, keys_2_entries, multiplicity_list
#     ):
#         # Retrieve the blocks
#         block_1 = tensor_1[key_1]
#         block_2 = tensor_2[key_2]

#         # Combine the blocks into a new TensorBlock of the correct lambda order.
#         # Pass the correction factor accounting for the redundancy of "lx"
#         # combinations.
#         nux_blocks.append(
#             _combine_single_center_block_pair(
#                 block_1,
#                 block_2,
#                 nux_key["spherical_harmonics_l"],
#                 cg_cache,
#                 correction_factor=np.sqrt(multi),
#             )
#         )

#     return TensorMap(nux_keys, nux_blocks)




# # Parse the angular and parity selections. Basic checks are performed here
    # # and errors raised if invalid selections are passed.
    # parity_selection = _parse_parity_selection(target_body_order, parity_selection)
    # angular_selection = _parse_angular_selection(
    #     target_body_order, rascal_hypers["max_angular"], angular_selection, angular_cutoff
    # )
    # if debug:
    #     print("parity_selection: ", parity_selection)
    #     print("angular_selection: ", angular_selection)
# # Pre-compute all the information needed to combined tensors at every
    # # iteration. This includes the keys of the TensorMaps produced at each
    # # iteration, the keys of the blocks combined to make them, and block
    # # multiplicities.
    # combine_info = []
    # for iteration in range(1, target_body_order):
    #     info = _create_combined_keys(
    #         nux_keys,
    #         nu1_keys,
    #         angular_selection[iteration - 1],
    #         parity_selection[iteration - 1],
    #     )
    #     combine_info.append(info)
    #     nux_keys = info[0]

    # if debug:
    #     print("Num. keys at each step: ", [len(c[0]) for c in combine_info])
    #     print([nu1_keys] + [c[0] for c in combine_info])
    #     return

    # if np.any([len(c[0]) == 0 for c in combine_info]):
    #     raise ValueError(
    #         "invalid filters: one or more iterations produce no valid combinations."
    #         f" Number of keys at each iteration: {[len(c[0]) for c in combine_info]}."
    #         " Check the `angular_selection` and `parity_selection` arguments."
    #     )
