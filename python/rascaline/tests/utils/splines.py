import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from rascaline.utils import RadialIntegralFromFunction


def sine(n: int, ell: int, positions: np.ndarray) -> np.ndarray:
    return np.sin(positions)


def cosine(n: int, ell: int, positions: np.ndarray) -> np.ndarray:
    return np.cos(positions)


@pytest.mark.parametrize("n_spline_points", [None, 1234])
def test_splines_with_n_spline_points(n_spline_points):
    spline_cutoff = 8.0

    spliner = RadialIntegralFromFunction(
        radial_integral=sine,
        max_radial=12,
        max_angular=9,
        spline_cutoff=spline_cutoff,
        radial_integral_derivative=cosine,
    )

    radial_integral = spliner.compute(n_spline_points=n_spline_points)[
        "TabulatedRadialIntegral"
    ]

    # check central contribution is not added
    with pytest.raises(KeyError):
        radial_integral["center_contribution"]

    spline_points = radial_integral["points"]

    # check that the first spline point is at 0
    assert spline_points[0]["position"] == 0.0

    # check that the last spline point is at the cutoff radius
    assert spline_points[-1]["position"] == 8.0

    # ensure correct length for values representation
    assert len(spline_points[52]["values"]["data"]) == (9 + 1) * 12

    # ensure correct length for derivatives representation
    assert len(spline_points[23]["derivatives"]["data"]) == (9 + 1) * 12

    # check values at r = 0.0
    assert np.allclose(
        np.array(spline_points[0]["values"]["data"]), np.zeros((9 + 1) * 12)
    )

    # check derivatives at r = 0.0
    assert np.allclose(
        np.array(spline_points[0]["derivatives"]["data"]), np.ones((9 + 1) * 12)
    )

    n_spline_points = len(spline_points)
    random_spline_point = 123
    random_x = random_spline_point * spline_cutoff / (n_spline_points - 1)

    # check value of a random spline point
    assert np.allclose(
        np.array(spline_points[random_spline_point]["values"]["data"]),
        np.sin(random_x) * np.ones((9 + 1) * 12),
    )


def test_splines_numerical_derivative():
    kwargs = {
        "radial_integral": sine,
        "max_radial": 12,
        "max_angular": 9,
        "spline_cutoff": 8.0,
    }

    spliner = RadialIntegralFromFunction(**kwargs, radial_integral_derivative=cosine)
    spliner_numerical = RadialIntegralFromFunction(**kwargs)

    spline_points = spliner.compute()["TabulatedRadialIntegral"]["points"]
    spline_points_numerical = spliner_numerical.compute()["TabulatedRadialIntegral"][
        "points"
    ]

    for s, s_num in zip(spline_points, spline_points_numerical):
        assert_equal(s["values"]["data"], s_num["values"]["data"])
        assert_allclose(
            s["derivatives"]["data"], s_num["derivatives"]["data"], rtol=1e-7
        )


def test_splines_numerical_derivative_error():
    kwargs = {
        "radial_integral": sine,
        "max_radial": 12,
        "max_angular": 9,
        "spline_cutoff": 1e-3,
    }

    match = "Numerically derivative of the radial integral can not be performed"
    with pytest.raises(ValueError, match=match):
        RadialIntegralFromFunction(**kwargs).compute()
