import ase
import gzip
import json
import numpy as np


def save_data(path, frames, hyperparameters, descriptor):
    frames = validate_frames(frames)

    with open(f"{path}-input.json", "w") as fd:
        json.dump(
            {
                "systems": [frame_to_json(f) for f in frames],
                "hyperparameters": hyperparameters,
            },
            fd,
            sort_keys=True,
            indent=2,
        )

    with gzip.open(f"{path}-values.npy.gz", "w") as fd:
        np.save(fd, descriptor.values)


def validate_frames(frames):
    if isinstance(frames, ase.Atoms):
        return [frames]
    else:
        assert isinstance(frames, list)
        for frame in frames:
            assert isinstance(frame, ase.Atoms)
        return frames


def frame_to_json(frame):
    return {
        "positions": frame.positions.tolist(),
        "species": frame.numbers.tolist(),
        "cell": np.concatenate(frame.cell).tolist(),
    }
