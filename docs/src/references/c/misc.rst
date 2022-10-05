Miscelaneous
============

Error handling
--------------

.. doxygenfunction:: rascal_last_error

.. doxygentypedef:: rascal_status_t

.. doxygendefine:: RASCAL_SUCCESS

.. doxygendefine:: RASCAL_INVALID_PARAMETER_ERROR

.. doxygendefine:: RASCAL_JSON_ERROR

.. doxygendefine:: RASCAL_UTF8_ERROR

.. doxygendefine:: RASCAL_CHEMFILES_ERROR

.. doxygendefine:: RASCAL_SYSTEM_ERROR

.. doxygendefine:: RASCAL_INTERNAL_ERROR

.. _c-api-logging:

Logging
-------

.. doxygenfunction:: rascal_set_logging_callback

.. doxygentypedef:: rascal_logging_callback_t

.. doxygendefine:: RASCAL_LOG_LEVEL_ERROR

.. doxygendefine:: RASCAL_LOG_LEVEL_WARN

.. doxygendefine:: RASCAL_LOG_LEVEL_INFO

.. doxygendefine:: RASCAL_LOG_LEVEL_DEBUG

.. doxygendefine:: RASCAL_LOG_LEVEL_TRACE

Profiling
---------

.. doxygenfunction:: rascal_profiling_enable

.. doxygenfunction:: rascal_profiling_clear

.. doxygenfunction:: rascal_profiling_get
