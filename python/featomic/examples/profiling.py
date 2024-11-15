"""
Profiling calculation
=====================

.. start-body
"""

import chemfiles

import featomic
from featomic import SoapPowerSpectrum


def compute_soap(path):
    """Compute SOAP power spectrum.

    This is the same code as the 'compute-soap' example
    """
    with chemfiles.Trajectory(path) as trajectory:
        frames = [f for f in trajectory]

    HYPER_PARAMETERS = {
        "cutoff": {
            "radius": 5.0,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 4,
            "radial": {"type": "Gto", "max_radial": 6},
        },
    }

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    descriptor = calculator.compute(frames, gradients=["positions"])
    descriptor = descriptor.keys_to_samples("center_type")
    descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

    return descriptor


# %%
#
# Run the calculation with profiling enabled.

with featomic.Profiler() as profiler:
    descriptor = compute_soap("dataset.xyz")
# %%
#
# Display the recorded profiling data as table.

print(profiler.as_short_table())

# %%
#
# You can also save this data as json for future usage
print(profiler.as_json())

# %%
#
# .. end-body
