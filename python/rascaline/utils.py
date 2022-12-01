# -*- coding: utf-8 -*-
import ctypes

from ._c_api import RASCAL_BUFFER_SIZE_ERROR
from .status import RascalError

import warnings


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except RascalError as e:
            if e.status == RASCAL_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def convert_old_hyperparameter_names(hyperparameters, mode):
    """
    Function to convert old hyperparameter names to those
    used in rascaline. This function is meant to be dep-
    recated as rascaline becomes more mainstream, but will
    serve to help users convert existing workflows.

    Notes
    -----
    - This function does validate the values in the hyperparameter
    dictionary, and it is up to the user to check that they pass
    valid entries to `rascaline`.
    - Not all of these parameters are supported in rascaline,
    and some will raise warnings.

    Parameters
    ----------

    mode: string in ["librascal", "dscribe"]
          We anticipate future support for mode=="quip" as well

    hyperparameters: dictionary of hyperparameter keys and values.
        For mode = `librascal`, the anticipated values are:
            - `coefficient_subselection`
            - `compute_gradients`
            - `covariant_lambda`
            - `cutoff_function_parameters`
            - `cutoff_function_type`
            - `cutoff_smooth_width`
            - `expansion_by_species_method`
            - `gaussian_sigma_constant`
            - `gaussian_sigma_type`
            - `global_species`
            - `interaction_cutoff`
            - `inversion_symmetry`
            - `max_angular`
            - `max_radial`
            - `normalize`
            - `optimization_args`
            - `optimization`
            - `radial_basis`
            - `soap_type`
        For mode = `dscribe`, the anticipated values are:
            - `average`
            - `crossover`
            - `dtype`
            - `nmax`
            - `lmax`
            - `periodic`
            - `rbf`
            - `rcut`
            - `sigma`
            - `sparse`
            - `species`

    """
    new_hypers = {}

    if mode == "librascal":
        anticipated_hypers = [
            "coefficient_subselection",
            "compute_gradients",
            "covariant_lambda",
            "cutoff_function_parameters",
            "cutoff_function_type",
            "cutoff_smooth_width",
            "expansion_by_species_method",
            "gaussian_sigma_constant",
            "gaussian_sigma_type",
            "global_species",
            "interaction_cutoff",
            "inversion_symmetry",
            "max_angular",
            "max_radial",
            "normalize",
            "optimization_args",
            "optimization",
            "radial_basis",
            "soap_type",
        ]

        if any([key not in anticipated_hypers for key in hyperparameters]):
            raise ValueError(
                "I do not know what to do with the following hyperparameter entries:\n\t".format(
                    "\n\t".join(
                        [
                            key
                            for key in hyperparameters
                            if key not in anticipated_hypers
                        ]
                    )
                )
            )

        new_hypers["atomic_gaussian_width"] = hyperparameters.pop(
            "gaussian_sigma_constant", None
        )
        new_hypers["max_angular"] = hyperparameters.pop("max_angular", None)
        new_hypers["max_radial"] = hyperparameters.pop("max_radial", None)
        new_hypers["cutoff"] = hyperparameters.pop("interaction_cutoff", None)

        if  'radial_basis' in hyperparameters:
            new_hypers["radial_basis"] = {hyperparameters.pop("radial_basis").title(): {}}
            if new_hypers["radial_basis"] != "Gto":
                warnings.warn("WARNING: rascaline currently only supports a Gto basis.")

        if hyperparameters.get("cutoff_function_type", None) == "ShiftedCosine":
            new_hypers["cutoff_function"] = {
                hyperparameters.pop("cutoff_function_type", None): {
                    "width": hyperparameters.pop("cutoff_smooth_width", None)
                }
            }
        else:
            new_hypers["cutoff_function"] = {"Step": {}}
            if hyperparameters.get("cutoff_function_type", None) == "RadialScaling":
                params = hyperparameters.pop("cutoff_function_parameters", None)
                new_hypers["radial_scaling"] = {
                    "Willatt2018": {
                        "exponent": int(params.get("exponent", None)),
                        "rate": params.get("rate", None),
                        "scale": params.get("scale", None),
                    }
                }

        deprecated_params = [
            "global_species",
            "expansion_by_species_method",
            "soap_type",
            "compute_gradients",
        ]
        if any([d in hyperparameters for d in deprecated_params]):
            warnings.warn(
                "{} are not required parameters in the rascaline software infrastructure".format(
                    ",".join(
                        [f"`{d}`" for d in deprecated_params if d in hyperparameters]
                    )
                )
            )

        not_supported = [
            "coefficient_subselection",
            "covariant_lambda",
            "gaussian_sigma_type",
            "inversion_symmetry",
            "normalize",
            "optimization_args",
            "optimization",
        ]
        if any([d in hyperparameters for d in not_supported]):
            warnings.warn(
                "{} are not currently supported in rascaline".format(
                    ",".join([f"`{d}`" for d in not_supported if d in hyperparameters])
                )
            )

        return {k: v for k,v in new_hypers.items() if v is not None}
    elif mode == "dscribe":
        anticipated_hypers = [
            "rcut",
            "nmax",
            "lmax",
            "species",
            "sigma",
            "rbf",
            "periodic",
            "crossover",
            "average",
            "sparse",
            "dtype",
        ]

        if any([key not in anticipated_hypers for key in hyperparameters]):
            raise ValueError(
                "I do not know what to do with the following hyperparameter entries:\n\t".format(
                    "\n\t".join(
                        [
                            key
                            for key in hyperparameters
                            if key not in anticipated_hypers
                        ]
                    )
                )
            )

        new_hypers["atomic_gaussian_width"] = hyperparameters.pop("sigma", None)
        new_hypers["max_angular"] = hyperparameters.pop("lmax", None)
        new_hypers["max_radial"] = hyperparameters.pop("nmax", None)
        new_hypers["cutoff"] = hyperparameters.pop("rcut", None)

        if  'rbf' in hyperparameters:
            new_hypers["radial_basis"] = {hyperparameters.pop("rbf").title(): {}}
            if new_hypers["radial_basis"] != "Gto":
                warnings.warn("WARNING: rascaline currently only supports a Gto basis.")

        deprecated_params = ["average", "sparse", "dtype"]
        if any([d in hyperparameters for d in deprecated_params]):
            warnings.warn(
                "{} are not required parameters in the rascaline software infrastructure".format(
                    ",".join(
                        [f"`{d}`" for d in deprecated_params if d in hyperparameters]
                    )
                )
            )

        not_supported = [
            "periodic",
            "crossover",
        ]
        if any([d in hyperparameters for d in not_supported]):
            warnings.warn(
                "{} are not currently supported in rascaline".format(
                    ",".join([f"`{d}`" for d in not_supported if d in hyperparameters])
                )
            )

        return {k: v for k,v in new_hypers.items() if v is not None}
    else:
        raise ValueError(
            f"Mode {mode} is not supported and must be either `librascal` or `dscribe`."
        )
