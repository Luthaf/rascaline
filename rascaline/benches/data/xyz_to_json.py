import json
from chemfiles import Trajectory
import numpy as np


def frame_to_json(frame):
    return {
        "cell": frame.cell.matrix.tolist(),
        "positions": frame.positions.tolist(),
        "species": [a.atomic_number for a in frame.atoms],
    }


def convert(input, output):
    data = []
    with Trajectory(input) as trajectory:
        for frame in trajectory:
            data.append(frame_to_json(frame))

    with open(output, "w") as fd:
        json.dump(data, fd, indent=2, sort_keys=True)


convert("silicon_bulk.xyz", "silicon_bulk.json")
convert("molecular_crystals.xyz", "molecular_crystals.json")
