def convert_soap(hypers):
    """convert from old style rascaline hypers for SOAP-related representations"""
    cleaned = {
        "cutoff": _process_cutoff(hypers),
        "density": _process_density(hypers),
    }

    max_angular = _get_or_error(hypers, "max_angular", "<root>")
    radial, spline_accuracy = _process_radial_basis(hypers)
    cleaned["basis"] = {
        "type": "TensorProduct",
        "max_angular": max_angular,
        "radial": radial,
    }
    if spline_accuracy is not None:
        if isinstance(spline_accuracy, float):
            cleaned["basis"]["spline_accuracy"] = spline_accuracy
        else:
            cleaned["basis"]["spline_accuracy"] = None

    return cleaned


def convert_radial_spectrum(hypers):
    """convert from old style rascaline hypers for SOAP radial spectrum"""
    cleaned = {
        "cutoff": _process_cutoff(hypers),
        "density": _process_density(hypers),
    }

    radial, spline_accuracy = _process_radial_basis(hypers)
    cleaned["basis"] = {"radial": radial}
    if spline_accuracy is not None:
        if isinstance(spline_accuracy, float):
            cleaned["basis"]["spline_accuracy"] = spline_accuracy
        else:
            cleaned["basis"]["spline_accuracy"] = None

    return cleaned


def convert_lode(hypers):
    """convert from old style rascaline hypers for LODE spherical expansion"""

    cleaned = {
        "density": _process_density(hypers),
    }

    max_angular = _get_or_error(hypers, "max_angular", "<root>")
    radial, spline_accuracy = _process_radial_basis(hypers, lode=True)
    cleaned["basis"] = {
        "type": "TensorProduct",
        "max_angular": max_angular,
        "radial": radial,
    }
    if spline_accuracy is not None:
        if isinstance(spline_accuracy, float):
            cleaned["basis"]["spline_accuracy"] = spline_accuracy
        else:
            cleaned["basis"]["spline_accuracy"] = None

    k_cutoff = hypers.get("k_cutoff")
    if k_cutoff is not None:
        cleaned["k_cutoff"] = k_cutoff

    return cleaned


def _process_cutoff(hypers):
    cutoff = {
        "radius": _get_or_error(hypers, "cutoff", "<root>"),
    }

    cutoff_fn = _get_or_error(hypers, "cutoff_function", "<root>")
    if "Step" in cutoff_fn:
        cutoff["smoothing"] = {"type": "Step"}
    if "ShiftedCosine" in cutoff_fn:
        width = _get_or_error(
            cutoff_fn["ShiftedCosine"], "width", "cutoff_function.ShiftedCosine"
        )
        cutoff["smoothing"] = {"type": "ShiftedCosine", "width": width}

    return cutoff


def _process_density(hypers):
    gaussian_width = _get_or_error(hypers, "atomic_gaussian_width", "<root>")
    center_weight = _get_or_error(hypers, "center_atom_weight", "<root>")
    exponent = hypers.get("potential_exponent")

    if exponent is None:
        density = {
            "type": "Gaussian",
            "width": gaussian_width,
        }
    else:
        density = {
            "type": "SmearedPowerLaw",
            "smearing": gaussian_width,
            "exponent": exponent,
        }

    if center_weight != 1.0:
        density["center_atom_weight"] = center_weight

    if "radial_scaling" in hypers:
        radial_scaling = hypers["radial_scaling"]
        if radial_scaling is None:
            pass

        if "None" in radial_scaling:
            pass

        if "Willatt2018" in radial_scaling:
            exponent = _get_or_error(
                radial_scaling["Willatt2018"], "exponent", "radial_scaling.Willatt2018"
            )
            rate = _get_or_error(
                radial_scaling["Willatt2018"], "rate", "radial_scaling.Willatt2018"
            )
            scale = _get_or_error(
                radial_scaling["Willatt2018"], "scale", "radial_scaling.Willatt2018"
            )

            density["scaling"] = {
                "type": "Willatt2018",
                "exponent": exponent,
                "rate": rate,
                "scale": scale,
            }

    return density


def _process_radial_basis(hypers, lode=False):
    spline_accuracy = None
    max_radial = _get_or_error(hypers, "max_radial", "<root>") - 1
    radial_basis = _get_or_error(hypers, "radial_basis", "<root>")

    if "Gto" in radial_basis:
        radial = {"type": "Gto", "max_radial": max_radial}

        if lode:
            cutoff = _get_or_error(hypers, "cutoff", "<root>") - 1
            radial["radius"] = cutoff

        gto_basis = radial_basis["Gto"]
        do_splines = gto_basis.get("splined_radial_integral", True)
        if do_splines:
            spline_accuracy = gto_basis.get("spline_accuracy")
        else:
            spline_accuracy = False

    elif "TabulatedRadialIntegral" in radial_basis:
        raise NotImplementedError("TabulatedRadialIntegral radial basis")

    return radial, spline_accuracy


def _get_or_error(hypers, name, path):
    from . import BadHyperParameters

    if name not in hypers:
        raise BadHyperParameters(f"missing {name} at {path} in hypers")

    return hypers.pop(name)
