import glob
import os
import re
import sys
from collections import namedtuple

import metatensor.torch
import torch

import featomic

from ._build_versions import BUILD_FEATOMIC_VERSION


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def version_compatible(actual, required):
    actual = parse_version(actual)
    required = parse_version(required)

    if actual.major != required.major:
        return False
    elif actual.minor != required.minor:
        return False
    else:
        return True


if not version_compatible(featomic.__version__, BUILD_FEATOMIC_VERSION):
    raise ImportError(
        f"Trying to load featomic-torch with featomic v{featomic.__version__}, "
        f"but it was compiled against featomic v{BUILD_FEATOMIC_VERSION}, which "
        "is not ABI compatible"
    )

_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    torch_version = parse_version(torch.__version__)
    install_prefix = os.path.join(
        _HERE, f"torch-{torch_version.major}.{torch_version.minor}"
    )

    if os.path.exists(install_prefix):
        if sys.platform.startswith("darwin"):
            path = os.path.join(install_prefix, "lib", "libfeatomic_torch.dylib")
            windows = False
        elif sys.platform.startswith("linux"):
            path = os.path.join(install_prefix, "lib", "libfeatomic_torch.so")
            windows = False
        elif sys.platform.startswith("win"):
            path = os.path.join(install_prefix, "bin", "featomic_torch.dll")
            windows = True
        else:
            raise ImportError("Unknown platform. Please edit this file")

        if os.path.isfile(path):
            if windows:
                _check_dll(path)
            return path
        else:
            raise ImportError("Could not find featomic_torch shared library at " + path)

    # gather which torch version(s) the current install was built
    # with to create the error message
    existing_versions = []
    for prefix in glob.glob(os.path.join(_HERE, "torch-*")):
        existing_versions.append(os.path.basename(prefix)[6:])

    if len(existing_versions) == 1:
        raise ImportError(
            f"Trying to load featomic-torch with torch v{torch.__version__}, "
            f"but it was compiled against torch v{existing_versions[0]}, which "
            "is not ABI compatible"
        )
    else:
        all_versions = ", ".join(map(lambda version: f"v{version}", existing_versions))
        raise ImportError(
            f"Trying to load featomic-torch with torch v{torch.__version__}, "
            f"we found builds for torch {all_versions}; which are not ABI compatible.\n"
            "You can try to re-install from source with "
            "`pip install featomic-torch --no-binary=featomic-torch`"
        )


def _check_dll(path):
    """
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    """
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but this DLL is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but this DLL is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


def _load_library():
    # Load featomic & metatensor-torch shared library in the process first, to ensure
    # the featomic_torch shared library can find them
    metatensor.torch._c_lib._load_library()

    featomic._c_lib._get_library()

    # load the C++ operators and custom classes
    torch.ops.load_library(_lib_path())
