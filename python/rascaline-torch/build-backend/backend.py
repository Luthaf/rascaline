# this is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to rascaline, using the local version if it exists, and otherwise
# falling back to the one on PyPI.
import os
import uuid

from setuptools import build_meta


ROOT = os.path.realpath(os.path.dirname(__file__))
RASCALINE = os.path.realpath(os.path.join(ROOT, "..", "..", ".."))
if os.path.exists(os.path.join(RASCALINE, "rascaline-c-api")):
    # we are building from a git checkout

    # add a random uuid to the file url to prevent pip from using a cached
    # wheel for metatensor-core, and force it to re-build from scratch
    uuid = uuid.uuid4()
    RASCALINE_DEP = f"rascaline @ file://{RASCALINE}?{uuid}"
else:
    # we are building from a sdist
    RASCALINE_DEP = "rascaline >=0.1.0.dev0,<0.2.0"


prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [
        "torch >= 1.12",
        "metatensor-torch >=0.4.0,<0.5.0",
        RASCALINE_DEP,
    ]


def get_requires_for_build_sdist(config_settings=None):
    defaults = build_meta.get_requires_for_build_sdist(config_settings)
    return defaults + [RASCALINE_DEP]
