from numpy.testing import assert_allclose

import rascaline
from rascaline.systems import IntoSystem


def finite_differences_positions(
    calculator: rascaline.CalculatorBase,
    system: IntoSystem,
    displacement: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-16,
) -> None:
    """
    Check that analytical gradients with respect to positions agree with a finite
    difference calculation of the gradients.

    The implementation is simular to ``rascaline/src/calculators/tests_utils.rs``.

    :param calculator: calculator used to compute the representation
    :param system: Atoms object
    :param displacement: distance each atom will be displaced in each direction when
        computing finite differences
    :param max_relative: Maximal relative error. ``10 * displacement`` is a good
        starting point
    :param atol: Threshold below which all values are considered zero. This should be
        very small (1e-16) to prevent false positives (if all values & gradients are
        below that threshold, tests will pass even with wrong gradients)
    :raises AssertionError: if the two gradients are not equal up to specified precision
    """
    reference = calculator.compute(system, gradients=["positions"])

    for atom_i in range(len(system)):
        for spatial in range(3):
            system_pos = system.copy()
            system_pos.positions[atom_i, spatial] += displacement / 2
            updated_pos = calculator.compute(system_pos)

            system_neg = system.copy()
            system_neg.positions[atom_i, spatial] -= displacement / 2
            updated_neg = calculator.compute(system_neg)

            assert updated_pos.keys == reference.keys
            assert updated_neg.keys == reference.keys

            for key, block in reference.items():
                gradients = block.gradient("positions")

                block_pos = updated_pos.block(key)
                block_neg = updated_neg.block(key)

                for gradient_i, (sample_i, _, atom) in enumerate(gradients.samples):
                    if atom != atom_i:
                        continue

                    # check that the sample is the same in both descriptors
                    assert block_pos.samples[sample_i] == block.samples[sample_i]
                    assert block_neg.samples[sample_i] == block.samples[sample_i]

                    value_pos = block_pos.values[sample_i]
                    value_neg = block_neg.values[sample_i]
                    gradient = gradients.values[gradient_i, spatial]

                    assert value_pos.shape == gradient.shape
                    assert value_neg.shape == gradient.shape

                    finite_difference = (value_pos - value_neg) / displacement

                    assert_allclose(finite_difference, gradient, rtol=rtol, atol=atol)
