import json
import re

from . import _rascaline


class BadHyperParameters(Exception):
    pass


def convert_hypers(origin, representation=None, hypers=None):
    """Convert hyper-parameters from other software into the format used by rascaline.

    :param origin: which software do the hyper-parameters come from? Valid values are:

        - ``"rascaline"`` for old rascaline format;
    :param representation: which representation are these hyper for? The meaning depend
        on the ``origin``:

        - for ``origin="rascaline"``, this is the name of the calculator class;
    :param hypers: the hyper parameter to convert. The type depend on the ``origin``:

        - for ``origin="rascaline"``, this should be a dictionary;

    :return: A string containing the code corresponding to the requested representation
        and hypers
    """
    if origin == "rascaline":
        if representation in [
            "SphericalExpansion",
            "SphericalExpansionByPair",
            "SoapPowerSpectrum",
        ]:
            hypers = _rascaline.convert_soap(hypers)
        elif representation == "SoapRadialSpectrum":
            hypers = _rascaline.convert_radial_spectrum(hypers)
        elif representation == "LodeSphericalExpansion":
            hypers = _rascaline.convert_lode(hypers)
        else:
            raise ValueError(
                "no hyper conversion exists for rascaline representation "
                f"'{representation}'"
            )

        hypers_dict = json.dumps(hypers, indent=4)
        hypers_dict = re.sub(r"\bnull\b", "None", hypers_dict)
        return f"{representation}(**{hypers_dict})"
    else:
        raise ValueError(f"no hyper conversion exists for {origin} software")
