Python and C interface
======================

How is the C interface exported
-------------------------------

Rascaline exports a C interface, defined in ``rascaline-c-api``. This C
interface is created directly in Rust, without involving any C code.

This is done by marking functions as ``#[no_mangle] extern pub fn <XXX>`` in
``rascaline-c-api``, and only using types safe to send to C (mostly pointers and
basic values such as floats or integers). Of these markers, ``pub`` ensures that
the function is exported from the library (it should appear as a ``T`` symbol in
``nm`` output); ``extern`` forces the function to use the C calling convention
(a calling convention describes where in memory/CPU registers the caller should
put data that the function expects); and ``#[no_mangle]`` tells the compiler to
export the function under this exact name, instead of using a mangled named
containing the module path and functions parameters.

Additionally, the C interfaces expose C-compatible structs declared with
``#[repr(C)] pub struct <XXX> {}``; where ``#[repr(C)]`` ensures that the
compiler lays out the fields in the exact order they are declared, without
re-organizing them.

``rascaline-c-api`` is then compiled to a shared library (``librascaline.so`` /
``librascaline.dylib`` / ``librascaline.dll``), which can be used by any
language able to call C code to call the exported functions without ever
realising it is speaking with Rust code.

The list of exported functions, together with the types of the functions
parameters, and struct definitions are extracted from the rust source code using
`cbindgen`_, which creates the ``rascaline-c-api/rascaline.h`` header file
containing all of this information in a C compatible syntax. All of the
documentation is also reproduced using `doxygen`_ syntax.


How does the Python interface works
-----------------------------------

The Python interface used the `ctypes`_ module to call exported symbols from the
shared library. For the Python code to be able to call exported function safely,
it needs to know a few things. In particular, it needs to know the name of the
function, the number and types of parameters and the return type of the
function. All this information is available in ``rascaline-c-api/rascaline.h``,
but not in a way that is easily accessible from `ctypes`_. There is a script in
``python/scripts/generate-declaration.py`` which reads the header file using
`pycparser`_, and creates the `python/rascaline/_rascaline.py` file which
declares all functions in the way expected by the `ctypes`_ module. You will
need to manually re-run this script if you modify any of the exported functions
in `rascaline-c-api`.

The schematic below describes all the relationships between the components
involved in creating the Python interface.

.. figure:: ../../static/images/rascaline-python.*
    :width: 400px
    :align: center

    Schematic representation of all components in the Python interface. The rust
    crate ``rascaline-c-api`` is compiled to a shared library
    (``librascaline.so`` on Linux), and `cbindgen`_ is used to generate the
    corresponding header. This header is then read with `pycparser`_ to create
    ctypes' compatible declarations, used to ensure that Python and rust agree
    fully on the parameters types, allowing Python to directly call Rust code.

.. _ctypes: https://docs.python.org/3/library/ctypes.html
.. _pycparser: https://github.com/eliben/pycparser
.. _cbindgen: https://github.com/eqrion/cbindgen/blob/master/docs.md
.. _doxygen: https://doxygen.org
