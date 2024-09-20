import pytest

from rascaline.calculators import (
    LodeSphericalExpansion,
    SoapPowerSpectrum,
    SoapRadialSpectrum,
    SphericalExpansion,
    SphericalExpansionByPair,
)


@pytest.mark.parametrize(
    "CalculatorClass",
    [
        SphericalExpansion,
        SphericalExpansionByPair,
        SoapPowerSpectrum,
        SoapRadialSpectrum,
    ],
)
def test_soap_hypers(CalculatorClass):
    message = (
        "hyper parameter changed recently, please update your code. "
        "Here are the new equivalent parameters"
    )
    with pytest.raises(ValueError, match=message) as err:
        CalculatorClass(
            atomic_gaussian_width=0.3,
            center_atom_weight=1.0,
            cutoff=3.4,
            cutoff_function={"ShiftedCosine": {"width": 0.5}},
            max_angular=5,
            max_radial=3,
            radial_basis={"Gto": {"spline_accuracy": 1e-3}},
            radial_scaling={"Willatt2018": {"exponent": 3, "rate": 2.2, "scale": 1.1}},
        )

    error_message = str(err.value.args[0])
    first_line = error_message.find("\n")
    code = error_message[first_line:]

    # max radial meaning changed
    assert '"max_radial": 2' in code

    # check that the error message contains valid code that can be copy/pasted
    eval(code)


def test_lode_hypers():
    message = (
        "hyper parameter changed recently, please update your code. "
        "Here are the new equivalent parameters"
    )
    with pytest.raises(ValueError, match=message) as err:
        LodeSphericalExpansion(
            atomic_gaussian_width=0.3,
            center_atom_weight=0.5,
            cutoff=3.4,
            cutoff_function={"Step": {}},
            max_angular=5,
            max_radial=3,
            radial_basis={"Gto": {"splined_radial_integral": False}},
            potential_exponent=3,
            k_cutoff=26.2,
        )

    error_message = str(err.value.args[0])
    first_line = error_message.find("\n")
    code = error_message[first_line:]

    # max radial meaning changed
    assert '"max_radial": 2' in code

    # check that the error message contains valid code that can be copy/pasted
    eval(code)
