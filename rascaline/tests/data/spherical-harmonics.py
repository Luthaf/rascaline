import math

import numpy as np
from save_data import save_json, save_numpy_array
from scipy.special import sph_harm


def real_sph(l, m, theta, phi):  # noqa: E741
    """Compute real spherical harmonics from the complex version in scipy"""
    m_1_pow_m = (-1) ** m
    if m > 0:
        return m_1_pow_m * np.real(sph_harm(m, l, theta, phi))
    elif m == 0:
        return m_1_pow_m * np.real(sph_harm(0, l, theta, phi)) / np.sqrt(2)
    else:
        return m_1_pow_m * np.imag(sph_harm(abs(m), l, theta, phi))


def spherical_harmonics(max_angular, directions):
    n_directions = len(directions)
    values = np.zeros(
        (n_directions, max_angular + 1, 2 * max_angular + 1), dtype=np.float64
    )

    phi = []
    theta = []
    for i_direction, direction in enumerate(directions):
        phi = math.acos(direction[2])
        theta = math.atan2(direction[1], direction[0])
        for l in range(max_angular + 1):  # noqa: E741
            for (i_m, m) in enumerate(range(-l, l + 1)):
                values[i_direction, l, i_m] = real_sph(l, m, theta, phi)

    return values


directions = [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.5773502691896258, 0.5773502691896258, 0.5773502691896258]),
    np.array([0.455493902781557, 0.46164788218724867, -0.7611875835829522]),
    np.array([-0.28695413002732584, -0.33058712239743676, -0.8990937002144119]),
    np.array([0.35108584385989333, -0.9226014654045358, -0.15982886558625636]),
]

parameters = {
    "max_angular": 25,
    "directions": directions,
}
values = spherical_harmonics(**parameters)

parameters["directions"] = [d.tolist() for d in parameters["directions"]]
save_json("spherical-harmonics", parameters)

save_numpy_array("spherical-harmonics-values", values, digits=13)
