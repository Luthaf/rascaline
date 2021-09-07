Profiling calculation
=====================

It can be interesting to know where a calculation is spending its time. To this
end, rascaline includes self-profiling code that can record and display which
part of the calculation takes time, and which function called long-running
functions. All the example should output something similar to the table below.

.. code-block:: text

    ╔════╦══════════════════════════════╦════════════╦═══════════╦══════════╦═════════════╗
    ║ id ║ span name                    ║ call count ║ called by ║ total    ║ mean        ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  1 ║ Calculator::compute          ║          2 ║         6 ║  18.45ms ║      9.23ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  6 ║ SoapPowerSpectrum::compute   ║          1 ║         1 ║   9.74ms ║      9.74ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  0 ║ NeighborsList                ║          1 ║         1 ║ 146.63µs ║    146.63µs ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  2 ║ Calculator::prepare          ║          2 ║         1 ║   3.48ms ║      1.74ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  4 ║ SphericalExpansion::compute  ║          1 ║         1 ║   4.65ms ║      4.65ms ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  3 ║ GtoRadialIntegral::compute   ║        299 ║         4 ║   1.76ms ║      5.88µs ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  5 ║ SphericalHarmonics::compute  ║        299 ║         4 ║ 193.73µs ║ 647.00ns ⚠️  ║
    ╠════╬══════════════════════════════╬════════════╬═══════════╬══════════╬═════════════╣
    ║  7 ║ Descriptor::densify          ║          1 ║         — ║   8.44ms ║      8.44ms ║
    ╚════╩══════════════════════════════╩════════════╩═══════════╩══════════╩═════════════╝

In this table, the first columns assign a unique numeric identifier to each
section of the code. The second one displays the name of the section. Then come
the number of time this section of the code have been executed, which other
function/section called the current one, and finally the total and mean time
spent in this function.

The ⚠️ symbol is added when the mean cost of the function is close to the
profiling overhead (around 100ns per function call), and thus the measurement
might not be very reliable.

Some of the most important sections are:

- ``Calculator::compute``: the entry point of all calculations
- ``Calculator::prepare``: building the list of samples that will be in the descriptor
- ``XXX::compute``: building blocks for the overall calculation
- ``NeighborsList``: construction of the list of neighbors

.. tabs::

    .. group-tab:: Python

        .. literalinclude:: ../../../python/examples/profiling.py
            :language: python

    .. group-tab:: Rust

        .. literalinclude:: ../../../rascaline/examples/profiling.rs
            :language: rust

    .. group-tab:: C++

        .. literalinclude:: ../../../rascaline-c-api/examples/profiling.cpp
            :language: c++

    .. group-tab:: C

        .. literalinclude:: ../../../rascaline-c-api/examples/profiling.c
            :language: c
