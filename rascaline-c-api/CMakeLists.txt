# Basic CMake integration for rascaline.
cmake_minimum_required(VERSION 3.10)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # Timestamp for FetchContent
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "" FORCE)

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" RASCALINE_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be rascaline version
        break()
    endif()
endforeach()

project(rascaline
    VERSION ${RASCALINE_VERSION}
    LANGUAGES C CXX
)

option(RASCALINE_DISABLE_CHEMFILES "Disable the usage of chemfiles for reading structures from files" OFF)
option(RASCALINE_FETCH_EQUISTORE "Download and build the equistore C API before building rascaline" OFF)

option(BUILD_SHARED_LIBS "Build a shared library instead of a static one" ON)
set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")
set(RUST_FLAGS "" CACHE STRING "additional rustc arguments used when compiling rascaline and dependencies")

mark_as_advanced(RUST_BUILD_TARGET RUST_FLAGS)


set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")


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

if (${RASCALINE_DISABLE_CHEMFILES})
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--no-default-features")
endif()

# Handle cross compilation with RUST_BUILD_TARGET
if (NOT "${RUST_BUILD_TARGET}" STREQUAL "")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
endif()

find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
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

# get the target directory from cargo. This relies on the JSON fields order not
# to change, which migth break. If this happen, we can build a small rust JSON
# parser to extract this info and give it back to cmake
execute_process(
    COMMAND ${CARGO_EXE} "metadata" "--no-deps" "--format-version" "1"
    RESULT_VARIABLE CARGO_METADATA_STATUS
    OUTPUT_VARIABLE CARGO_METADATA
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
if(CARGO_METADATA_STATUS AND NOT CARGO_METADATA_STATUS EQUAL 0)
    message(FATAL_ERROR "could not run cargo metadata")
endif()
string(REGEX REPLACE ".*\"target_directory\":\"(.*)\",\"version.*" "\\1/${CARGO_BUILD_TYPE}" CARGO_TARGET_DIR "${CARGO_METADATA}")

# get the parent directory to remove debug/release from the path. We add it back
# just below, but cargo metadata does not account for a different build target.
get_filename_component(CARGO_TARGET_DIR ${CARGO_TARGET_DIR} DIRECTORY)
if ("${RUST_BUILD_TARGET}" STREQUAL "")
    set(CARGO_TARGET_DIR "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}")
else()
    set(CARGO_TARGET_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")
endif()

if(${BUILD_SHARED_LIBS})
    add_library(rascaline SHARED IMPORTED GLOBAL)
    set(RASCALINE_CAPI_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rascaline_c_api${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(RASCALINE_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rascaline${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    add_library(rascaline STATIC IMPORTED GLOBAL)
    set(RASCALINE_CAPI_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rascaline_c_api${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(RASCALINE_LOCATION "${CARGO_TARGET_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rascaline${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

get_filename_component(RASCALINE_LIB_NAME ${RASCALINE_LOCATION} NAME)

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path. This is used for rascaline_torch python's
# package amongst other.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/librascaline.dylib`
        set(SONAME_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${RASCALINE_LIB_NAME}")
    else() # LINUX
        # set the SONAME to librascaline.so, and the RPATH to $ORIGIN
        set(SONAME_ARGS "-Clink-arg=-Wl,-soname,${RASCALINE_LIB_NAME},-rpath,\$ORIGIN")
    endif()
else()
    set(SONAME_ARGS "")
endif()

if (RASCAL_BUILD_FOR_PYTHON)
    # strip dynamic library for smaller wheels to download/install
    set(STRIP_ARGS "-Cstrip=symbols")
else()
    set(STRIP_ARGS "")
endif()

if (RASCALINE_BUILD_FOR_PYTHON)
    # strip dynamic library for smaller wheels to download/install
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};-Cstrip=symbols")
endif()

add_custom_target(cargo-build-rascaline ALL
    COMMAND ${CMAKE_COMMAND} -E env
        MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        RUSTFLAGS=${RUST_FLAGS}
        cargo rustc ${CARGO_BUILD_ARG} -- ${SONAME_ARGS} ${STRIP_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${RASCALINE_LIB_NAME} with cargo"
    VERBATIM
)

add_custom_command(TARGET cargo-build-rascaline
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${RASCALINE_CAPI_LOCATION} ${RASCALINE_LOCATION}
    DEPENDS cargo-build-rascaline
)

add_dependencies(rascaline cargo-build-rascaline)
set(RASCALINE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/rascaline.h"
    "${PROJECT_SOURCE_DIR}/include/rascaline.hpp"
)
set(RASCALINE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set(EQUISTORE_GIT_VERSION "6c5a833")
set(EQUISTORE_REQUIRED_VERSION "0.1")
if (RASCALINE_FETCH_EQUISTORE)
    message(STATUS "Fetching equistore @ ${EQUISTORE_GIT_VERSION} from github")

    if (CMAKE_VERSION VERSION_LESS 3.11)
        # cmake 3.10 (default on ubuntu 20.04) does not include FetchContent,
        # but we can use the version from cmake 3.11 without change
        set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch-content-cmake-3.11;${CMAKE_MODULE_PATH}")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        equistore
        URL      https://github.com/lab-cosmo/equistore/archive/${EQUISTORE_GIT_VERSION}.zip
        URL_HASH MD5=5adcd26f236022ea89fa0287b83c6609
        SOURCE_SUBDIR equistore-core
        VERBOSE
    )

    if (CMAKE_VERSION VERSION_GREATER 3.18)
        FetchContent_MakeAvailable(equistore)
    else()
        if (NOT equistore_POPULATED)
            FetchContent_Populate(equistore)
        endif()

        add_subdirectory(${equistore_SOURCE_DIR}/equistore-core ${equistore_BINARY_DIR})
    endif()

    add_dependencies(cargo-build-rascaline equistore)
else()
    find_package(equistore ${EQUISTORE_REQUIRED_VERSION} REQUIRED CONFIG)
endif()

set_target_properties(rascaline PROPERTIES
    IMPORTED_LOCATION ${RASCALINE_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE_DIR}
)
target_link_libraries(rascaline INTERFACE equistore)

if(NOT ${RASCALINE_DISABLE_CHEMFILES})
    # the static library will need to be linked as C++ code since it contains
    # chemfiles
    set_target_properties(rascaline PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
    )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ${BUILD_SHARED_LIBS})
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    # the rust standard lib uses pthread and libdl on linux
    target_link_libraries(rascaline INTERFACE Threads::Threads dl)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

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
    DESTINATION ${LIB_INSTALL_DIR}/cmake/rascaline
)
