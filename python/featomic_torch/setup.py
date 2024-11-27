import glob
import os
import subprocess
import sys

import packaging
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel


ROOT = os.path.realpath(os.path.dirname(__file__))

FEATOMIC_PYTHON_SRC = os.path.realpath(os.path.join(ROOT, "..", "featomic"))
FEATOMIC_TORCH_SRC = os.path.realpath(os.path.join(ROOT, "..", "..", "featomic-torch"))


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        import metatensor
        import metatensor.torch
        import torch

        import featomic

        source_dir = FEATOMIC_TORCH_SRC
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "featomic/torch")

        os.makedirs(build_dir, exist_ok=True)

        # Tell CMake where to find featomic & torch
        cmake_prefix_path = [
            featomic.utils.cmake_prefix_path,
            metatensor.utils.cmake_prefix_path,
            metatensor.torch.utils.cmake_prefix_path,
            torch.utils.cmake_prefix_path,
        ]

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_PREFIX_PATH={';'.join(cmake_prefix_path)}",
        ]

        # ==================================================================== #
        # HACK: Torch cmake build system has a hard time finding CuDNN, so we
        # help it by pointing it to the right files

        # First try using the `nvidia.cudnn` package (dependency of torch on PyPI)
        try:
            import nvidia.cudnn

            cudnn_root = os.path.dirname(nvidia.cudnn.__file__)
        except ImportError:
            # Otherwise try to find CuDNN inside PyTorch itself
            cudnn_root = os.path.join(torch.utils.cmake_prefix_path, "..", "..")

            cudnn_version = os.path.join(cudnn_root, "include", "cudnn_version.h")
            if not os.path.exists(cudnn_version):
                # create a minimal cudnn_version.h (with a made-up version),
                # because it is not bundled together with the CuDNN shared
                # library in PyTorch conda distribution, see
                # https://github.com/pytorch/pytorch/issues/47743
                with open(cudnn_version, "w") as fd:
                    fd.write("#define CUDNN_MAJOR 8\n")
                    fd.write("#define CUDNN_MINOR 5\n")
                    fd.write("#define CUDNN_PATCHLEVEL 0\n")

        cmake_options.append(f"-DCUDNN_INCLUDE_DIR={cudnn_root}/include")
        cmake_options.append(f"-DCUDNN_LIBRARY={cudnn_root}/lib")
        # do not warn if the two variables above aren't used
        cmake_options.append("--no-warn-unused-cli")

        # end of HACK
        # ==================================================================== #

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                build_dir,
                "--parallel",
                "--config",
                "Release",
                "--target",
                "install",
            ],
            check=True,
        )

        with open(os.path.join(install_dir, "_build_versions.py"), "w") as fd:
            fd.write("# Autogenerated file, do not edit\n\n\n")
            # Store the version of torch used to build the extension, to give a
            # nice error message to the user when trying to load the extension
            # with an older torch version installed
            fd.write(
                "# version of torch used when compiling this package\n"
                f"BUILD_TORCH_VERSION = '{torch.__version__}'\n\n"
            )

            # same for featomic
            fd.write(
                "# version of featomic used when compiling this package\n"
                f"BUILD_FEATOMIC_VERSION = '{featomic.__version__}'\n"
            )


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "install dist/metatensor-*.whl` to install from source."
        )


class sdist_generate_data(sdist):
    """
    Create a sdist with an additional generated files:
        - `git_version_info`
        - `featomic-torch-cxx-*.tar.gz`
    """

    def run(self):
        n_commits, git_hash = git_version_info()
        with open("git_version_info", "w") as fd:
            fd.write(f"{n_commits}\n{git_hash}\n")

        generate_cxx_tar()

        # run original sdist
        super().run()

        os.unlink("git_version_info")
        for path in glob.glob("featomic-torch-cxx-*.tar.gz"):
            os.unlink(path)


def generate_cxx_tar():
    script = os.path.join(ROOT, "..", "..", "scripts", "package-featomic-torch.sh")
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


def git_version_info():
    """
    If git is available and we are building from a checkout, get the number of commits
    since the last tag & full hash of the code. Otherwise, this always returns (0, "").
    """
    TAG_PREFIX = "featomic-torch-v"

    if os.path.exists("git_version_info"):
        # we are building from a sdist, without git available, but the git
        # version was recorded in the `git_version_info` file
        with open("git_version_info") as fd:
            n_commits = int(fd.readline().strip())
            git_hash = fd.readline().strip()
    else:
        script = os.path.join(ROOT, "..", "..", "scripts", "git-version-info.py")
        assert os.path.exists(script)

        output = subprocess.run(
            [sys.executable, script, TAG_PREFIX],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf8",
        )

        if output.returncode != 0:
            raise Exception(
                "failed to get git version info.\n"
                f"stdout: {output.stdout}\n"
                f"stderr: {output.stderr}\n"
            )
        elif output.stderr:
            print(output.stderr, file=sys.stderr)
            n_commits = 0
            git_hash = ""
        else:
            lines = output.stdout.splitlines()
            n_commits = int(lines[0].strip())
            git_hash = lines[1].strip()

    return n_commits, git_hash


def create_version_number(version):
    version = packaging.version.parse(version)

    n_commits, git_hash = git_version_info()
    if n_commits != 0:
        # `n_commits` will be non zero only if we have commits since the last tag. This
        # mean we are in a pre-release of the next version. So we increase either the
        # minor version number or the release candidate number (if we are closing up on
        # a release)
        if version.pre is not None:
            assert version.pre[0] == "rc"
            pre = ("rc", version.pre[1] + 1)
            release = version.release
        else:
            major, minor, patch = version.release
            release = (major, minor + 1, 0)
            pre = None

        # this is using a private API which is intended to become public soon:
        # https://github.com/pypa/packaging/pull/698. In the mean time we'll
        # use this
        version._version = version._version._replace(release=release)
        version._version = version._version._replace(pre=pre)
        version._version = version._version._replace(dev=("dev", n_commits))
        version._version = version._version._replace(local=(git_hash,))

    return str(version)


if __name__ == "__main__":
    if not os.path.exists(FEATOMIC_TORCH_SRC):
        # we are building from a sdist, which should include featomic-torch
        # sources as a tarball
        tarballs = glob.glob(os.path.join(ROOT, "featomic-torch-cxx-*.tar.gz"))

        if not len(tarballs) == 1:
            raise RuntimeError(
                "expected a single 'featomic-torch-cxx-*.tar.gz' file containing "
                "featomic-torch C++ sources"
            )

        FEATOMIC_TORCH_SRC = os.path.realpath(tarballs[0])
        subprocess.run(
            ["cmake", "-E", "tar", "xf", FEATOMIC_TORCH_SRC],
            cwd=ROOT,
            check=True,
        )

        FEATOMIC_TORCH_SRC = ".".join(FEATOMIC_TORCH_SRC.split(".")[:-2])

    with open(os.path.join(FEATOMIC_TORCH_SRC, "VERSION")) as fd:
        version = create_version_number(fd.read().strip())

    with open(os.path.join(ROOT, "AUTHORS")) as fd:
        authors = fd.read().splitlines()

    if authors[0].startswith(".."):
        # handle "raw" symlink files (on Windows or from full repo tarball)
        with open(os.path.join(ROOT, authors[0])) as fd:
            authors = fd.read().splitlines()

    install_requires = [
        "torch >= 1.12",
        "metatensor-torch >=0.6.0,<0.7.0",
    ]
    if os.path.exists(FEATOMIC_PYTHON_SRC):
        # we are building from a git checkout
        install_requires.append(f"featomic @ file://{FEATOMIC_PYTHON_SRC}")
    else:
        # we are building from a sdist/installing from a wheel
        install_requires.append("featomic >=0.1.0.dev0,<0.2.0")

    setup(
        version=version,
        author=", ".join(authors),
        install_requires=install_requires,
        ext_modules=[
            Extension(name="featomic_torch", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
            "sdist": sdist_generate_data,
        },
        package_data={
            "featomic-torch": [
                "featomic/torch/bin/*",
                "featomic/torch/lib/*",
                "featomic/torch/include/*",
            ]
        },
    )
