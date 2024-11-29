from setuptools import build_meta


get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [
        "cmake",
        "metatensor-core >=0.1.11,<0.2.0",
    ]
