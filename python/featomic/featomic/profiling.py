from ._c_lib import _get_library
from .calculator_base import _call_with_growing_buffer


class Profiler:
    """Profiler recording execution time of featomic functions.

    Featomic uses the `time_graph <https://docs.rs/time-graph/>`_ to collect
    timing information on the calculations. The ``Profiler`` class can be used
    as a context manager to access to this functionality.

    The profiling code collects the total time spent inside the most important
    functions, as well as the function call graph (which function called which
    other function).

    .. code-block:: python

        import featomic

        with featomic.Profiler() as profiler:
            # run some calculations
            ...

        print(profiler.as_short_table())
    """

    def __init__(self):
        self._lib = _get_library()

    def __enter__(self):
        self._lib.featomic_profiling_enable(True)
        self._lib.featomic_profiling_clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lib.featomic_profiling_enable(False)

    def as_json(self):
        """Get current profiling data formatted as JSON."""
        return _call_with_growing_buffer(
            lambda b, s: self._lib.featomic_profiling_get("json".encode("utf8"), b, s)
        )

    def as_table(self):
        """Get current profiling data formatted as a table."""
        return _call_with_growing_buffer(
            lambda b, s: self._lib.featomic_profiling_get("table".encode("utf8"), b, s)
        )

    def as_short_table(self):
        """
        Get current profiling data formatted as a table, using short functions names.
        """
        return _call_with_growing_buffer(
            lambda b, s: self._lib.featomic_profiling_get(
                "short_table".encode("utf8"), b, s
            )
        )
