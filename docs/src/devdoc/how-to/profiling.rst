Profiling calculation
=====================

It can be interesting to know where a calculation is spending its time. To this
end, featomic includes self-profiling code that can record and display which
part of the calculation takes time, and which function called long-running
functions. All the example should output something similar to the table below.

.. code-block:: text

    ╔════╦══════════════════════════════╦════════════╦═══════════╦══════════╦═════════════╗
    ║ id ║ span name                    ║ call count ║ called by ║ total    ║ mean        ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  2 ║ Full calculation             ║          1 ║         — ║ 660.58ms ║    660.58ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  3 ║ SoapPowerSpectrum::compute   ║          1 ║         2 ║ 584.02ms ║    584.02ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  1 ║ Calculator::prepare          ║          2 ║      3, 2 ║ 148.15ms ║     74.08ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  0 ║ NeighborsList                ║         20 ║         1 ║  20.82ms ║      1.04ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  5 ║ SphericalExpansion::compute  ║          1 ║         3 ║ 196.38ms ║    196.38ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  4 ║ GtoRadialIntegral::compute   ║      74448 ║         5 ║ 117.04ms ║      1.57µs ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  6 ║ SphericalHarmonics::compute  ║      74448 ║         5 ║   9.95ms ║ 133.00ns ⚠️ ║
    ╚════╩══════════════════════════════╩════════════╩═══════════╩══════════╩═════════════╝

In this table, the first columns assign a unique numeric identifier to each
section of the code. The second one displays the name of the section. Then come
the number of time this section of the code have been executed, which other
function/section called the current one, and finally the total and mean time
spent in this function.

The ⚠️ symbol is added when the mean cost of the function is close to the
profiling overhead (30 to 80ns per function call), and thus the measurement
might not be very reliable.

Some of the most important sections are:

- ``Calculator::prepare``: building the list of samples/properties that will be in the descriptor
- ``XXX::compute``: building blocks for the overall calculation
- ``NeighborsList``: construction of the list of neighbors

You can obtain a dataset for profiling from our :download:`website <../../../static/dataset.xyz>`.

.. tabs::

    .. group-tab:: Python

        .. container:: sphx-glr-footer sphx-glr-footer-example

            .. container:: sphx-glr-download sphx-glr-download-python

                :download:`Download Python source code for this example: profiling.py <../../examples/profiling.py>`

            .. container:: sphx-glr-download sphx-glr-download-jupyter

                :download:`Download Jupyter notebook for this example: profiling.ipynb <../../examples/profiling.ipynb>`

        .. include:: ../../examples/profiling.rst
            :start-after: start-body
            :end-before: end-body

    .. group-tab:: Rust

        .. literalinclude:: ../../../../featomic/examples/profiling.rs
            :language: rust

    .. group-tab:: C++

        .. literalinclude:: ../../../../featomic/examples/profiling.cpp
            :language: c++

    .. group-tab:: C

        .. literalinclude:: ../../../../featomic/examples/profiling.c
            :language: c
