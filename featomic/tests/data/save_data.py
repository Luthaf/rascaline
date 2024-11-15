import gzip
import io
import json
import math
import os

import ase
import numpy as np


ROOT = os.path.abspath(os.path.dirname(__file__))


def save_calculator_input(path, frames, hyperparameters):
    frames = validate_frames(frames)
    save_json(
        path,
        {
            "systems": [frame_to_json(f) for f in frames],
            "hyperparameters": hyperparameters,
        },
    )


def save_json(path, data):
    with open(os.path.join(ROOT, "generated", f"{path}-input.json"), "w") as fd:
        json.dump(
            data,
            fd,
            sort_keys=True,
            indent=2,
        )
        fd.write("\n")


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
        "types": frame.numbers.tolist(),
        "cell": np.concatenate(frame.cell).tolist(),
    }


def save_numpy_array(path, array, digits=10):
    """
    Save a numpy array as gzip compressed npy at the given path, making sure the
    content of the generated file only depends on the content of array.

    In  particular, this function removed the timestamp, filename and OS ID from
    the gzip header to ensure that re-generating the files will be as
    deterministic as possible.

    The array values are also rounded to ``digits`` significant digits to remove
    small differences related to floating point environments.
    """
    if digits is not None:
        array = np.vectorize(lambda x: signif(x, digits))(array)

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as fd:
        np.save(fd, array)

    # byte 9 is an operating system identifier, set it to the UNKNOWN value
    # see https://docs.fileformat.com/compression/gz/
    view = buffer.getbuffer()
    view[9] = 0xFF

    with open(os.path.join(ROOT, "generated", f"{path}.npy.gz"), "wb") as fd:
        fd.write(view)


def signif(x, digits):
    """
    Round x to the request number of significant digits
    """
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)
