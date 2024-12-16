Miscellaneous
=============

Error handling
--------------

.. doxygenfunction:: featomic_last_error

.. doxygentypedef:: featomic_status_t

.. doxygendefine:: FEATOMIC_SUCCESS

.. doxygendefine:: FEATOMIC_INVALID_PARAMETER_ERROR

.. doxygendefine:: FEATOMIC_JSON_ERROR

.. doxygendefine:: FEATOMIC_UTF8_ERROR

.. doxygendefine:: FEATOMIC_SYSTEM_ERROR

.. doxygendefine:: FEATOMIC_INTERNAL_ERROR

.. _c-api-logging:

Logging
-------

.. doxygenfunction:: featomic_set_logging_callback

.. doxygentypedef:: featomic_logging_callback_t

.. doxygendefine:: FEATOMIC_LOG_LEVEL_ERROR

.. doxygendefine:: FEATOMIC_LOG_LEVEL_WARN

.. doxygendefine:: FEATOMIC_LOG_LEVEL_INFO

.. doxygendefine:: FEATOMIC_LOG_LEVEL_DEBUG

.. doxygendefine:: FEATOMIC_LOG_LEVEL_TRACE

Profiling
---------

.. doxygenfunction:: featomic_profiling_enable

.. doxygenfunction:: featomic_profiling_clear

.. doxygenfunction:: featomic_profiling_get
