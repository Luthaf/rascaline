# Basic CMake integration for rascaline.
cmake_minimum_required(VERSION 3.10)

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
string(REGEX REPLACE ".*version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" RASCALINE_VERSION ${CARGO_TOML_CONTENT})

project(rascaline
    VERSION ${RASCALINE_VERSION}
    LANGUAGES C # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)

option(RASCAL_DISABLE_CHEMFILES "Disable the usage of chemfiles for reading structures from files" OFF)
option(BUILD_SHARED_LIBS "Build a shared library instead of a static one" ON)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

# TODO: support multiple configuration generators (MSVC, ...)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ("${BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_ARG "")
    set(CARGO_BUILD_TYPE "debug")
elseif("${BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "--release")
    set(CARGO_BUILD_TYPE "release")
elseif("${BUILD_TYPE}" STREQUAL "relwithdebinfo")
    set(CARGO_BUILD_ARG "--release")
    set(CARGO_BUILD_TYPE "release")
else()
    message(FATAL_ERROR "unsuported build type: ${CMAKE_BUILD_TYPE}")
endif()

if (${RASCAL_DISABLE_CHEMFILES})
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--no-default-features")
endif()

find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
    )
endif()

execute_process(
    COMMAND ${CARGO_EXE} "--version"
    RESULT_VARIABLE CARGO_STATUS
    OUTPUT_VARIABLE CARGO_VERSION
)

if(CARGO_STATUS AND NOT CARGO_STATUS EQUAL 0)
    message(FATAL_ERROR
        "could not run cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install)"
    )
endif()

set(REQUIRED_RUST_VERSION "1.42.0")
string(REGEX REPLACE "cargo ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" CARGO_VERSION ${CARGO_VERSION})
if (${CARGO_VERSION} VERSION_LESS ${REQUIRED_RUST_VERSION})
    message(FATAL_ERROR
        "your Rust installation is too old (you have version ${CARGO_VERSION}), \
        at least ${REQUIRED_RUST_VERSION} is required"
    )
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/../Cargo.toml
    ${PROJECT_SOURCE_DIR}/../rascaline/Cargo.toml
    ${PROJECT_SOURCE_DIR}/../rascaline/src/**.rs

    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/build.rs
    ${PROJECT_SOURCE_DIR}/src/**.rs
)

set(RASCALINE_LIBDIR "${PROJECT_SOURCE_DIR}/../target/${CARGO_BUILD_TYPE}/")
if(${BUILD_SHARED_LIBS})
    add_library(rascaline SHARED IMPORTED GLOBAL)
    set(RASCALINE_LOCATION "${RASCALINE_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rascaline${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    add_library(rascaline STATIC IMPORTED GLOBAL)
    set(RASCALINE_LOCATION "${RASCALINE_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rascaline${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

get_filename_component(RASCALINE_LIB_NAME ${RASCALINE_LOCATION} NAME)
add_custom_target(cargo-build-rascaline ALL
    COMMAND cargo build ${CARGO_BUILD_ARG}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${RASCALINE_LIB_NAME} with cargo"
)

add_dependencies(rascaline cargo-build-rascaline)
set(RASCALINE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/rascaline.h"
    "${PROJECT_SOURCE_DIR}/include/rascaline.hpp"
)
set(RASCALINE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(rascaline PROPERTIES
    IMPORTED_LOCATION ${RASCALINE_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE_DIR}
)

if(NOT ${RASCAL_DISABLE_CHEMFILES})
    # the static library will need to be linked as C++ code since it contains
    # chemfiles
    set_target_properties(rascaline PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
    )
endif()

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
    set_target_properties(rascaline PROPERTIES
        IMPORTED_LINK_INTERFACE_LIBRARIES "pthread;dl"
    )
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/rascaline-config.in.cmake"
    "${PROJECT_BINARY_DIR}/rascaline-config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/rascaline
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rascaline-config-version.in.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rascaline-config-version.cmake"
    @ONLY
)

install(FILES ${RASCALINE_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})
install(FILES ${RASCALINE_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
install(FILES
    ${PROJECT_BINARY_DIR}/rascaline-config-version.cmake
    ${PROJECT_BINARY_DIR}/rascaline-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/chemfiles
)
