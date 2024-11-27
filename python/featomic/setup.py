import glob
import os
import shutil
import subprocess
import sys
import uuid

from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


ROOT = os.path.realpath(os.path.dirname(__file__))
FEATOMIC_SRC = os.path.realpath(os.path.join(ROOT, "..", "..", "featomic"))

FEATOMIC_BUILD_TYPE = os.environ.get("FEATOMIC_BUILD_TYPE", "release")
if FEATOMIC_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{FEATOMIC_BUILD_TYPE}',"
        "expected 'debug' or 'release'"
    )

# The code for featomic-torch needs to live in `featomic_torch` directory until
# https://github.com/pypa/pip/issues/13093 is fixed.
FEATOMIC_TORCH_SRC = os.path.realpath(os.path.join(ROOT, "..", "featomic_torch"))


class universal_wheel(bdist_wheel):
    """Helper class for override wheel tag.

    When building the wheel, the `wheel` package assumes that if we have a
    binary extension then we are linking to `libpython.so`; and thus the wheel
    is only usable with a single python version. This is not the case for
    here, and the wheel will be compatible with any Python >=3.6. This is
    tracked in https://github.com/pypa/wheel/issues/185, but until then we
    manually override the wheel tag.
    """

    def get_tag(self):
        """Get the tag for override."""
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """Build the native library using cmake."""

    def run(self):
        """Run cmake build and install the resulting library."""
        source_dir = FEATOMIC_SRC
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "featomic")

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DCMAKE_BUILD_TYPE={FEATOMIC_BUILD_TYPE}",
            "-DFEATOMIC_FETCH_METATENSOR=ON",
            "-DFEATOMIC_INSTALL_BOTH_STATIC_SHARED=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DEXTRA_RUST_FLAGS=-Cstrip=symbols",
        ]

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        # Handle cross-compilation by detecting cibuildwheels environnement
        # variables
        if sys.platform.startswith("darwin"):
            # ARCHFLAGS is set by cibuildwheels
            ARCHFLAGS = os.environ.get("ARCHFLAGS")
            if ARCHFLAGS is not None:
                archs = filter(
                    lambda u: bool(u),
                    ARCHFLAGS.strip().split("-arch "),
                )
                archs = list(archs)
                assert len(archs) == 1
                arch = archs[0].strip()

                if arch == "x86_64":
                    cmake_options.append("-DRUST_BUILD_TARGET=x86_64-apple-darwin")
                elif arch == "arm64":
                    cmake_options.append("-DRUST_BUILD_TARGET=aarch64-apple-darwin")
                else:
                    raise ValueError(f"unknown arch: {arch}")

        elif sys.platform.startswith("linux"):
            # we set RUST_BUILD_TARGET in our custom docker image
            RUST_BUILD_TARGET = os.environ.get("RUST_BUILD_TARGET")
            if RUST_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={RUST_BUILD_TARGET}")

        elif sys.platform.startswith("win32"):
            # CARGO_BUILD_TARGET is set by cibuildwheels
            CARGO_BUILD_TARGET = os.environ.get("CARGO_BUILD_TARGET")
            if CARGO_BUILD_TARGET is not None:
                cmake_options.append(f"-DRUST_BUILD_TARGET={CARGO_BUILD_TARGET}")

        else:
            raise ValueError(f"unknown platform: {sys.platform}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--parallel", "--target", "install"],
            check=True,
        )

        # do not include metatensor libraries/headers/cmake config within
        # featomic wheel
        for file in glob.glob(os.path.join(install_dir, "lib", "libmetatensor.*")):
            os.unlink(file)

        for file in glob.glob(os.path.join(install_dir, "bin", "metatensor.dll")):
            os.unlink(file)

        shutil.rmtree(os.path.join(install_dir, "lib", "cmake", "metatensor"))

        for file in glob.glob(os.path.join(install_dir, "include", "metatensor*")):
            os.unlink(file)


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "uninstall metatensor -y && pip install dist/metatensor-*.whl` "
            + "to install from source."
        )


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `git_extra_version`
        - `featomic-cxx-*.tar.gz`
    """

    def run(self):
        with open("git_extra_version", "w") as fd:
            fd.write(git_extra_version())

        generate_cxx_tar()

        # run original sdist
        super().run()

        os.unlink("git_extra_version")
        for path in glob.glob("featomic-cxx-*.tar.gz"):
            os.unlink(path)


def generate_cxx_tar():
    script = os.path.join(ROOT, "..", "..", "scripts", "package-featomic.sh")
    assert os.path.exists(script)

    try:
        output = subprocess.run(
            ["bash", "--version"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
    except Exception as e:
        raise RuntimeError("could not run `bash`, is it installed?") from e

    output = subprocess.run(
        ["bash", script, os.getcwd()],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf8",
    )
    if output.returncode != 0:
        stderr = output.stderr
        stdout = output.stdout
        raise RuntimeError(
            "failed to collect C++ sources for Python sdist\n"
            f"stdout:\n {stdout}\n\nstderr:\n {stderr}"
        )


def get_rust_version():
    # read version from Cargo.toml
    with open(os.path.join(FEATOMIC_SRC, "Cargo.toml")) as fd:
        for line in fd:
            if line.startswith("version"):
                _, version = line.split(" = ")
                # remove quotes
                version = version[1:-2]
                # take the first version in the file, this should be the right
                # version
                break

    return version


def git_extra_version():
    """
    If git is available, it is used to check if we are installing a development
    version or a released version (by checking how many commits happened since
    the last tag).
    """

    # Add pre-release info the version
    try:
        tags_list = subprocess.run(
            ["git", "tag"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            check=True,
        )
        tags_list = tags_list.stdout.decode("utf8").strip()

        if tags_list == "":
            first_commit = subprocess.run(
                ["git", "rev-list", "--max-parents=0", "HEAD"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                check=True,
            )
            reference = first_commit.stdout.decode("utf8").strip()

        else:
            last_tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                check=True,
            )

            reference = last_tag.stdout.decode("utf8").strip()

    except Exception:
        reference = ""
        pass

    try:
        n_commits_since_tag = subprocess.run(
            ["git", "rev-list", f"{reference}..HEAD", "--count"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            check=True,
        )
        n_commits_since_tag = n_commits_since_tag.stdout.decode("utf8").strip()

        if n_commits_since_tag != 0:
            return ".dev" + n_commits_since_tag
    except Exception:
        pass

    return ""


if __name__ == "__main__":
    if not os.path.exists(FEATOMIC_SRC):
        # we are building from a sdist, which should include featomic Rust
        # sources as a tarball
        tarballs = glob.glob(os.path.join(ROOT, "featomic-*.tar.gz"))

        if not len(tarballs) == 1:
            raise RuntimeError(
                "expected a single 'featomic-*.tar.gz' file containing "
                "featomic Rust sources. remove all files and re-run "
                "scripts/package-featomic.sh"
            )

        FEATOMIC_SRC = os.path.realpath(tarballs[0])
        subprocess.run(
            ["cmake", "-E", "tar", "xf", FEATOMIC_SRC],
            cwd=ROOT,
            check=True,
        )

        FEATOMIC_SRC = ".".join(FEATOMIC_SRC.split(".")[:-2])

    if os.path.exists("git_extra_version"):
        # we are building from a sdist, without git available, but the git
        # version was recorded in a git_extra_version file
        with open("git_extra_version") as fd:
            extra_version = fd.read()
    else:
        extra_version = git_extra_version()

    version = get_rust_version() + extra_version

    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    extras_require = {}
    if os.path.exists(FEATOMIC_TORCH_SRC):
        # we are building from a git checkout

        # add a random uuid to the file url to prevent pip from using a cached
        # wheel for featomic-torch, and force it to re-build from scratch
        uuid = uuid.uuid4()
        extras_require["torch"] = f"featomic-torch @ file://{FEATOMIC_TORCH_SRC}?{uuid}"
    else:
        # we are building from a sdist/installing from a wheel
        extras_require["torch"] = "featomic-torch >=0.1.0.dev0,<0.2.0"

    setup(
        version=version,
        author=", ".join(authors),
        extras_require=extras_require,
        ext_modules=[
            # only declare the extension, it is built & copied as required by cmake
            # in the build_ext command
            Extension(name="featomic", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
            "sdist": sdist_generate_data,
        },
        package_data={
            "featomic": [
                "featomic/lib/*",
                "featomic/include/*",
            ]
        },
    )
