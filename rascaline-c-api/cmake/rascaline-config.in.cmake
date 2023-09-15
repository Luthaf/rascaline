@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.16)

if(rascaline_FOUND)
    return()
endif()

enable_language(CXX)

if (WIN32)
    set(RASCALINE_SHARED_LOCATION ${PACKAGE_PREFIX_DIR}/@BIN_INSTALL_DIR@/@RASCALINE_SHARED_LIB_NAME@)
    set(RASCALINE_IMPLIB_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_IMPLIB_NAME@)
else()
    set(RASCALINE_SHARED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_SHARED_LIB_NAME@)
endif()

set(RASCALINE_STATIC_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_STATIC_LIB_NAME@)
set(RASCALINE_INCLUDE ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/)

if (NOT EXISTS ${RASCALINE_INCLUDE}/rascaline.h OR NOT EXISTS ${RASCALINE_INCLUDE}/rascaline.hpp)
    message(FATAL_ERROR "could not find rascaline headers in '${RASCALINE_INCLUDE}', please re-install rascaline")
endif()

find_package(metatensor @METATENSOR_REQUIRED_VERSION@ REQUIRED CONFIG)

# Shared library target
if (@RASCALINE_INSTALL_BOTH_STATIC_SHARED@ OR @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${RASCALINE_SHARED_LOCATION})
        message(FATAL_ERROR "could not find rascaline library at '${RASCALINE_SHARED_LOCATION}', please re-install rascaline")
    endif()

    add_library(rascaline::shared SHARED IMPORTED GLOBAL)
    set_target_properties(rascaline::shared PROPERTIES
        IMPORTED_LOCATION ${RASCALINE_SHARED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE}
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
    )
    target_link_libraries(rascaline::shared INTERFACE metatensor::shared)

    target_compile_features(rascaline::shared INTERFACE cxx_std_17)

    if (WIN32)
        if (NOT EXISTS ${RASCALINE_IMPLIB_LOCATION})
            message(FATAL_ERROR "could not find rascaline library at '${RASCALINE_IMPLIB_LOCATION}', please re-install rascaline")
        endif()

        set_target_properties(rascaline::shared PROPERTIES
            IMPORTED_IMPLIB ${RASCALINE_IMPLIB_LOCATION}
        )
    endif()
endif()


# Static library target
if (@RASCALINE_INSTALL_BOTH_STATIC_SHARED@ OR NOT @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${RASCALINE_STATIC_LOCATION})
        message(FATAL_ERROR "could not find rascaline library at '${RASCALINE_STATIC_LOCATION}', please re-install rascaline")
    endif()

    add_library(rascaline::static STATIC IMPORTED GLOBAL)
    set_target_properties(rascaline::static PROPERTIES
        IMPORTED_LOCATION ${RASCALINE_STATIC_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE}
        INTERFACE_LINK_LIBRARIES "@CARGO_DEFAULT_LIBRARIES@"
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
    )
    target_link_libraries(rascaline::static INTERFACE metatensor::shared)

    target_compile_features(rascaline::static INTERFACE cxx_std_17)
endif()


# Export either the shared or static library as the rascaline target
if (@BUILD_SHARED_LIBS@)
    add_library(rascaline ALIAS rascaline::shared)
else()
    add_library(rascaline ALIAS rascaline::static)
endif()
