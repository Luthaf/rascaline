@PACKAGE_INIT@

cmake_minimum_required(VERSION 3.16)

if(featomic_FOUND)
    return()
endif()

enable_language(CXX)

get_filename_component(FEATOMIC_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/@PACKAGE_RELATIVE_PATH@" ABSOLUTE)

if (WIN32)
    set(FEATOMIC_SHARED_LOCATION ${FEATOMIC_PREFIX_DIR}/@BIN_INSTALL_DIR@/@FEATOMIC_SHARED_LIB_NAME@)
    set(FEATOMIC_IMPLIB_LOCATION ${FEATOMIC_PREFIX_DIR}/@LIB_INSTALL_DIR@/@FEATOMIC_IMPLIB_NAME@)
else()
    set(FEATOMIC_SHARED_LOCATION ${FEATOMIC_PREFIX_DIR}/@LIB_INSTALL_DIR@/@FEATOMIC_SHARED_LIB_NAME@)
endif()

set(FEATOMIC_STATIC_LOCATION ${FEATOMIC_PREFIX_DIR}/@LIB_INSTALL_DIR@/@FEATOMIC_STATIC_LIB_NAME@)
set(FEATOMIC_INCLUDE ${FEATOMIC_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/)

if (NOT EXISTS ${FEATOMIC_INCLUDE}/featomic.h OR NOT EXISTS ${FEATOMIC_INCLUDE}/featomic.hpp)
    message(FATAL_ERROR "could not find featomic headers in '${FEATOMIC_INCLUDE}', please re-install featomic")
endif()

find_package(metatensor @METATENSOR_REQUIRED_VERSION@ REQUIRED CONFIG)

# Shared library target
if (@FEATOMIC_INSTALL_BOTH_STATIC_SHARED@ OR @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${FEATOMIC_SHARED_LOCATION})
        message(FATAL_ERROR "could not find featomic library at '${FEATOMIC_SHARED_LOCATION}', please re-install featomic")
    endif()

    add_library(featomic::shared SHARED IMPORTED GLOBAL)
    set_target_properties(featomic::shared PROPERTIES
        IMPORTED_LOCATION ${FEATOMIC_SHARED_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${FEATOMIC_INCLUDE}
    )
    target_link_libraries(featomic::shared INTERFACE metatensor::shared)

    target_compile_features(featomic::shared INTERFACE cxx_std_17)

    if (WIN32)
        if (NOT EXISTS ${FEATOMIC_IMPLIB_LOCATION})
            message(FATAL_ERROR "could not find featomic library at '${FEATOMIC_IMPLIB_LOCATION}', please re-install featomic")
        endif()

        set_target_properties(featomic::shared PROPERTIES
            IMPORTED_IMPLIB ${FEATOMIC_IMPLIB_LOCATION}
        )
    endif()
endif()


# Static library target
if (@FEATOMIC_INSTALL_BOTH_STATIC_SHARED@ OR NOT @BUILD_SHARED_LIBS@)
    if (NOT EXISTS ${FEATOMIC_STATIC_LOCATION})
        message(FATAL_ERROR "could not find featomic library at '${FEATOMIC_STATIC_LOCATION}', please re-install featomic")
    endif()

    add_library(featomic::static STATIC IMPORTED GLOBAL)
    set_target_properties(featomic::static PROPERTIES
        IMPORTED_LOCATION ${FEATOMIC_STATIC_LOCATION}
        INTERFACE_INCLUDE_DIRECTORIES ${FEATOMIC_INCLUDE}
        INTERFACE_LINK_LIBRARIES "@CARGO_DEFAULT_LIBRARIES@"
    )
    target_link_libraries(featomic::static INTERFACE metatensor::shared)

    target_compile_features(featomic::static INTERFACE cxx_std_17)
endif()


# Export either the shared or static library as the featomic target
if (@BUILD_SHARED_LIBS@)
    add_library(featomic ALIAS featomic::shared)
else()
    add_library(featomic ALIAS featomic::static)
endif()
