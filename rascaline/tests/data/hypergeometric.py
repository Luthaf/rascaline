import numpy as np
import mpmath as math

from save_data import save_numpy_array, save_json


def gto_gaussian_constants(max_radial, cutoff=3.333333):
    constants = []
    for n in range(max_radial):
        sigma = cutoff * math.sqrt(max(n, 1)) / (max_radial + 1.0)
        constants.append(1.0 / (2.0 * sigma * sigma))
    return [float(c) for c in constants]


def spherical_hypergeometric(
    max_radial, max_angular, atomic_gaussian_constants, gto_gaussian_constants, all_rij
):
    # Compute everything up to 50 decimal places
    math.mp.dps = 50

    n_rij = len(all_rij)
    n_constants = len(atomic_gaussian_constants)
    values = np.zeros(
        (n_constants, n_rij, max_radial, max_angular + 1), dtype=np.float64
    )
    gradients = np.zeros(
        (n_constants, n_rij, max_radial, max_angular + 1), dtype=np.float64
    )

    for i_atomic_constant, c in enumerate(atomic_gaussian_constants):
        for i_rij, rij in enumerate(all_rij):
            for n in range(max_radial):
                for l in range(max_angular + 1):  # noqa: E741
                    a = (n + l + 3) / 2.0
                    b = l + 1.5
                    d = gto_gaussian_constants[n]

                    z = c * c * rij * rij / (c + d)

                    gamma_ratio = math.gamma(a) / math.gamma(b)

                    value = (
                        gamma_ratio * math.exp(-c * rij * rij) * math.hyp1f1(a, b, z)
                    )
                    values[i_atomic_constant, i_rij, n, l] = value

                    grad_factor = 2.0 * a * c * c * rij / (b * (c + d))
                    grad = (
                        grad_factor
                        * gamma_ratio
                        * math.exp(-c * rij * rij)
                        * math.hyp1f1(a + 1.0, b + 1.0, z)
                    )
                    gradients[i_atomic_constant, i_rij, n, l] = (
                        grad - 2.0 * c * rij * value
                    )

    return values, gradients


parameters = {
    "max_radial": 20,
    "max_angular": 20,
    "atomic_gaussian_constants": [0.1, 0.3535, 1.0, 3.0],
    "all_rij": [0.05, 0.2, 1.0, 4.0, 8.0, 10.0],
}
parameters["gto_gaussian_constants"] = gto_gaussian_constants(parameters["max_angular"])

save_json("hypergeometric", parameters)

values, gradients = spherical_hypergeometric(**parameters)
save_numpy_array("hypergeometric-values", values, digits=15)
save_numpy_array("hypergeometric-gradients", gradients, digits=15)
