import os
import shutil
import subprocess
import sys
from datetime import datetime

import toml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(os.path.join(ROOT, "docs", "extensions"))

os.environ["FEATOMIC_IMPORT_FOR_SPHINX"] = "1"
os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"

# -- Project information -----------------------------------------------------

project = "Featomic"
author = ", ".join(open(os.path.join(ROOT, "AUTHORS")).read().splitlines())
copyright = f"{datetime.now().date().year}, {author}"


def load_version_from_cargo_toml():
    with open(os.path.join(ROOT, "featomic", "Cargo.toml")) as fd:
        data = toml.load(fd)
    return data["package"]["version"]


# The full version, including alpha/beta/rc tags
release = load_version_from_cargo_toml()


def build_cargo_docs():
    environment = {name: value for name, value in os.environ.items()}

    # include KaTeX in the page to render math in the docs
    katex_html = os.path.join(ROOT, "docs", "inject-katex.html")
    environment["RUSTDOCFLAGS"] = f"--html-in-header={katex_html}"

    subprocess.run(
        [
            "cargo",
            "doc",
            "--package",
            "featomic",
            "--package",
            "metatensor",
            "--no-deps",
        ],
        env=environment,
    )
    output_dir = os.path.join(
        ROOT,
        "docs",
        "build",
        "html",
        "references",
        "api",
        "rust",
    )
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(
        os.path.join(ROOT, "target", "doc"),
        output_dir,
    )


def extract_json_schema():
    subprocess.run(["cargo", "run", "--package", "featomic-json-schema"])


def build_doxygen_docs():
    # we need to run a build to make sure the header is up to date
    subprocess.run(["cargo", "build", "--package", "featomic-c-api"])
    subprocess.run(["doxygen", "Doxyfile"], cwd=os.path.join(ROOT, "docs"))


extract_json_schema()
build_cargo_docs()
build_doxygen_docs()


def setup(app):
    app.add_css_file("featomic.css")


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "breathe",
    "sphinx_gallery.gen_gallery",
    "sphinx_tabs.tabs",
    "featomic_json_schema",
    "html_hidden",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "sg_execution_times.rst",
]


autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": ["../../python/featomic/examples"],
    "gallery_dirs": ["examples"],
    "min_reported_time": 5,
    # Make the code snippet for featomic functions clickable
    "reference_url": {"featomic": None},
    "prefer_full_module": ["featomic"],
}

breathe_projects = {
    "featomic": os.path.join(ROOT, "docs", "build", "doxygen", "xml"),
}
breathe_default_project = "featomic"
breathe_domain_by_extension = {
    "h": "c",
}

intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "chemfiles": ("https://chemfiles.org/chemfiles.py/latest/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skmatter": ("https://scikit-matter.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [os.path.join(ROOT, "docs", "static")]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/metatensor/featomic",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# font-awesome logos (used in the footer)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
