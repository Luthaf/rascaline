# Basic CMake integration for featomic.
cmake_minimum_required(VERSION 3.16)

# Is featomic the main project configured by the user? Or is this being used
# as a submodule/subdirectory?
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(FEATOMIC_MAIN_PROJECT ON)
else()
    set(FEATOMIC_MAIN_PROJECT OFF)
endif()

if(${FEATOMIC_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # Timestamp for FetchContent
endif()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # use variables to set OPTIONS
endif()

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" FEATOMIC_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be featomic version
        break()
    endif()
endforeach()

include(cmake/dev-versions.cmake)
create_development_version("${FEATOMIC_VERSION}" FEATOMIC_FULL_VERSION "featomic-v")
message(STATUS "Building featomic v${FEATOMIC_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" FEATOMIC_VERSION ${FEATOMIC_FULL_VERSION})
project(featomic
    VERSION ${FEATOMIC_VERSION}
    LANGUAGES C CXX # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)
set(PROJECT_VERSION ${FEATOMIC_FULL_VERSION})

# We follow the standard CMake convention of using BUILD_SHARED_LIBS to provide
# either a shared or static library as a default target. But since cargo always
# builds both versions by default, we also install both versions by default.
# `FEATOMIC_INSTALL_BOTH_STATIC_SHARED=OFF` allow to disable this behavior, and
# only install the file corresponding to `BUILD_SHARED_LIBS=ON/OFF`.
#
# BUILD_SHARED_LIBS controls the `featomic` cmake target, making it an alias of
# either `featomic::static` or `featomic::shared`. This is mainly relevant
# when using featomic from another cmake project, either as a submodule or from
# an installed library (see cmake/featomic-config.cmake)
option(BUILD_SHARED_LIBS "Use a shared library by default instead of a static one" ON)
option(FEATOMIC_INSTALL_BOTH_STATIC_SHARED "Install both shared and static libraries" ON)

set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")
set(EXTRA_RUST_FLAGS "" CACHE STRING "Flags used to build rust code")
mark_as_advanced(RUST_BUILD_TARGET EXTRA_RUST_FLAGS)

option(FEATOMIC_FETCH_METATENSOR "Download and build the metatensor C API before building featomic" OFF)

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")

if (${FEATOMIC_MAIN_PROJECT})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if(${FEATOMIC_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_BUILD_TYPE}" STREQUAL ${CMAKE_BUILD_TYPE})
    set(CACHED_LAST_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Last build type used in configuration")
    message(STATUS "Building featomic in ${CMAKE_BUILD_TYPE} mode")
endif()

find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
    )
endif()

execute_process(
    COMMAND ${CARGO_EXE} "--version" "--verbose"
    RESULT_VARIABLE CARGO_STATUS
    OUTPUT_VARIABLE CARGO_VERSION_RAW
)

if(CARGO_STATUS AND NOT CARGO_STATUS EQUAL 0)
    message(FATAL_ERROR
        "could not run cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install)"
    )
endif()

if (CARGO_VERSION_RAW MATCHES "cargo ([0-9]+\\.[0-9]+\\.[0-9]+).*")
    set(CARGO_VERSION "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "failed to determine cargo version, output was: ${CARGO_VERSION_RAW}")
endif()
if(NOT "${CACHED_LAST_CARGO_VERSION}" STREQUAL ${CARGO_VERSION})
    set(CACHED_LAST_CARGO_VERSION ${CARGO_VERSION} CACHE INTERNAL "Last version of cargo used in configuration")
    message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
    set(CARGO_VERSION_CHANGED TRUE)
endif()

# ============================================================================ #
# determine Cargo flags

set(CARGO_BUILD_ARG "")

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--locked")
endif()

# TODO: support multiple configuration generators (MSVC, ...)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ("${BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_TYPE "debug")
elseif("${BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
elseif("${BUILD_TYPE}" STREQUAL "relwithdebinfo")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
else()
    message(FATAL_ERROR "unsuported build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target-dir=${CARGO_TARGET_DIR}")
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--features=c-api")

# Handle cross compilation with RUST_BUILD_TARGET
if ("${RUST_BUILD_TARGET}" STREQUAL "")
        if (CARGO_VERSION_RAW MATCHES "host: ([a-zA-Z0-9_\\-]*)\n")
        set(RUST_HOST_TARGET "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "failed to determine host target, output was: ${CARGO_VERSION_RAW}")
    endif()

    if (${FEATOMIC_MAIN_PROJECT})
        message(STATUS "Compiling to host (${RUST_HOST_TARGET})")
    endif()

    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}")
    set(RUST_BUILD_TARGET ${RUST_HOST_TARGET})
else()
    if (${FEATOMIC_MAIN_PROJECT})
        message(STATUS "Cross-compiling to ${RUST_BUILD_TARGET}")
    endif()

    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")
endif()

# Get the list of libraries linked by default by cargo/rustc to add when linking
# to featomic::static
if (CARGO_VERSION_CHANGED)
    include(cmake/tempdir.cmake)
    get_tempdir(TMPDIR)

    # Adapted from https://github.com/corrosion-rs/corrosion/blob/dc1e4e5/cmake/FindRust.cmake
    execute_process(
        COMMAND "${CARGO_EXE}" new --lib _cargo_required_libs
        WORKING_DIRECTORY "${TMPDIR}"
        RESULT_VARIABLE cargo_new_result
        ERROR_QUIET
    )

    if (cargo_new_result)
        message(FATAL_ERROR "could not create empty project to find default static libs: ${cargo_new_result}")
    endif()

    file(APPEND "${TMPDIR}/_cargo_required_libs/Cargo.toml" "[lib]\ncrate-type=[\"staticlib\"]")

    execute_process(
        COMMAND ${CARGO_EXE} rustc --color never --target=${RUST_BUILD_TARGET} -- --print=native-static-libs
        WORKING_DIRECTORY "${TMPDIR}/_cargo_required_libs"
        RESULT_VARIABLE cargo_build_result
        ERROR_VARIABLE cargo_build_error_message
    )

    # clean up the files
    file(REMOVE_RECURSE "${TMPDIR}")

    if(cargo_build_result)
        message(FATAL_ERROR "could extract default static libs: ${cargo_build_result}")
    endif()

    # The pattern starts with `native-static-libs:` and goes to the end of the line.
    if(cargo_build_error_message MATCHES "native-static-libs: ([^\r\n]+)\r?\n")
        string(REPLACE " " ";" "libs_list" "${CMAKE_MATCH_1}")
        set(stripped_lib_list "")
        foreach(lib ${libs_list})
            # Strip leading `-l` (unix) and potential .lib suffix (windows)
            string(REGEX REPLACE "^-l" "" "stripped_lib" "${lib}")
            string(REGEX REPLACE "\.lib$" "" "stripped_lib" "${stripped_lib}")
            list(APPEND stripped_lib_list "${stripped_lib}")
        endforeach()
        # Special case `msvcrt` to link with the debug version in Debug mode.
        list(TRANSFORM stripped_lib_list REPLACE "^msvcrt$" "\$<\$<CONFIG:Debug>:msvcrtd>")
        list(REMOVE_DUPLICATES stripped_lib_list)

        set(CARGO_DEFAULT_LIBRARIES "${stripped_lib_list}" CACHE INTERNAL "list of implicitly linked libraries")
        if (${FEATOMIC_MAIN_PROJECT})
            message(STATUS "Cargo default link libraries are: ${CARGO_DEFAULT_LIBRARIES}")
        endif()
    else()
        message(FATAL_ERROR "could not find default static libs: `native-static-libs` not found in: `${cargo_build_error_message}`")
    endif()
endif()

# ============================================================================ #
# Setup metatensor

# METATENSOR_FETCH_VERSION is the exact version we will fetch from github if
# FEATOMIC_FETCH_METATENSOR=ON, and METATENSOR_REQUIRED_VERSION is the minimal
# version we require when using `find_package` to find the library.
#
# When updating METATENSOR_FETCH_VERSION, you will also have to update the
# SHA256 sum of the file in `FetchContent_Declare`.
set(METATENSOR_FETCH_VERSION "0.1.11")
set(METATENSOR_REQUIRED_VERSION "0.1")
if (FEATOMIC_FETCH_METATENSOR)
    message(STATUS "Fetching metatensor-core from github")

    include(FetchContent)
    set(URL_ROOT "https://github.com/lab-cosmo/metatensor/releases/download")
    FetchContent_Declare(
        metatensor
        URL      ${URL_ROOT}/metatensor-core-v${METATENSOR_FETCH_VERSION}/metatensor-core-cxx-${METATENSOR_FETCH_VERSION}.tar.gz
        URL_HASH SHA256=b79435dbeb59a95361b4a5881574dfde4a4ed13f12b05a225df161dcaa7ea61e
    )

    if (CMAKE_VERSION VERSION_GREATER 3.18)
        FetchContent_MakeAvailable(metatensor)
    else()
        if (NOT metatensor_POPULATED)
            FetchContent_Populate(metatensor)
        endif()

        add_subdirectory(${metatensor_SOURCE_DIR} ${metatensor_BINARY_DIR})
    endif()

    # metatensor will be installed in the same place as featomic, so set
    # the RPATH to ${ORIGIN} to load the file from there
    set(METATENSOR_RPATH "$$\\{ORIGIN\\}")
else()
    find_package(metatensor ${METATENSOR_REQUIRED_VERSION} REQUIRED CONFIG)
    # in case featomic gets installed in a different place than metatensor,
    # set the RPATH to the directory where we found metatensor
    get_target_property(METATENSOR_LOCATION metatensor::shared IMPORTED_LOCATION)
    get_filename_component(METATENSOR_LOCATION ${METATENSOR_LOCATION} DIRECTORY)
    set(METATENSOR_RPATH "${METATENSOR_LOCATION}")

    message(STATUS "Using local metatensor from ${METATENSOR_LOCATION}")
endif()

# ============================================================================ #
# Setup featomic libraries

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/../Cargo.toml
    ${PROJECT_SOURCE_DIR}/../featomic/Cargo.toml
    ${PROJECT_SOURCE_DIR}/../featomic/src/**.rs

    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/build.rs
    ${PROJECT_SOURCE_DIR}/src/**.rs
)

add_library(featomic::shared SHARED IMPORTED GLOBAL)
set(FEATOMIC_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}featomic${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(FEATOMIC_IMPLIB_LOCATION "${FEATOMIC_SHARED_LOCATION}.lib")

add_library(featomic::static STATIC IMPORTED GLOBAL)
set(FEATOMIC_STATIC_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}featomic${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(FEATOMIC_SHARED_LIB_NAME ${FEATOMIC_SHARED_LOCATION} NAME)
get_filename_component(FEATOMIC_IMPLIB_NAME     ${FEATOMIC_IMPLIB_LOCATION} NAME)
get_filename_component(FEATOMIC_STATIC_LIB_NAME ${FEATOMIC_STATIC_LOCATION} NAME)

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/libfeatomic.dylib`
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${FEATOMIC_SHARED_LIB_NAME}")
    else() # LINUX
        # set the SONAME to libfeatomic.so, and point the RPATH to metatensor
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-soname,${FEATOMIC_SHARED_LIB_NAME},-rpath=${METATENSOR_RPATH}")
    endif()
else()
    set(CARGO_RUSTC_ARGS "")
endif()

if (NOT "${EXTRA_RUST_FLAGS}" STREQUAL "")
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};${EXTRA_RUST_FLAGS}")
endif()

# Set environement variables for cargo build
set(CARGO_ENV "")
if (NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
    list(APPEND CARGO_ENV "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
if (NOT "$ENV{RUSTC_WRAPPER}" STREQUAL "")
    list(APPEND CARGO_ENV "RUSTC_WRAPPER=$ENV{RUSTC_WRAPPER}")
endif()

if (FEATOMIC_INSTALL_BOTH_STATIC_SHARED)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib;--crate-type=staticlib")
    set(FILES_CREATED_BY_CARGO "${FEATOMIC_SHARED_LIB_NAME} and ${FEATOMIC_STATIC_LIB_NAME}")
else()
    if (BUILD_SHARED_LIBS)
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib")
        set(FILES_CREATED_BY_CARGO "${FEATOMIC_SHARED_LIB_NAME}")
    else()
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=staticlib")
        set(FILES_CREATED_BY_CARGO "${FEATOMIC_STATIC_LIB_NAME}")
    endif()
endif()

add_custom_target(cargo-build-featomic ALL
    COMMAND
        ${CMAKE_COMMAND} -E env ${CARGO_ENV}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${FILES_CREATED_BY_CARGO} with cargo"
    BYPRODUCTS ${FEATOMIC_STATIC_LOCATION} ${FEATOMIC_SHARED_LOCATION} ${FEATOMIC_IMPLIB_LOCATION}
)

add_dependencies(featomic::shared cargo-build-featomic)
add_dependencies(featomic::static cargo-build-featomic)

set(FEATOMIC_HEADERS
    "${PROJECT_SOURCE_DIR}/include/featomic.h"
    "${PROJECT_SOURCE_DIR}/include/featomic.hpp"
)
set(FEATOMIC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(featomic::shared PROPERTIES
    IMPORTED_LOCATION ${FEATOMIC_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${FEATOMIC_INCLUDE_DIR}
    BUILD_VERSION "${FEATOMIC_FULL_VERSION}"
)
target_compile_features(featomic::shared INTERFACE cxx_std_17)

if (WIN32)
    set_target_properties(featomic::shared PROPERTIES
        IMPORTED_IMPLIB ${FEATOMIC_IMPLIB_LOCATION}
    )
endif()

set_target_properties(featomic::static PROPERTIES
    IMPORTED_LOCATION ${FEATOMIC_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${FEATOMIC_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "${CARGO_DEFAULT_LIBRARIES}"
    BUILD_VERSION "${FEATOMIC_FULL_VERSION}"
)

if (BUILD_SHARED_LIBS)
    add_library(featomic ALIAS featomic::shared)
else()
    add_library(featomic ALIAS featomic::static)
endif()

add_dependencies(cargo-build-featomic metatensor)
target_link_libraries(featomic::shared INTERFACE metatensor::shared)
target_link_libraries(featomic::static INTERFACE metatensor::shared)

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/featomic-config.in.cmake"
    "${PROJECT_BINARY_DIR}/featomic-config.cmake"
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/featomic
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/featomic-config-version.in.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/featomic-config-version.cmake"
    @ONLY
)

install(FILES ${FEATOMIC_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})

if (FEATOMIC_INSTALL_BOTH_STATIC_SHARED OR BUILD_SHARED_LIBS)
    if (WIN32)
        # DLL files should go in <prefix>/bin
        install(FILES ${FEATOMIC_SHARED_LOCATION} DESTINATION ${BIN_INSTALL_DIR})
        # .lib files should go in <prefix>/lib
        install(FILES ${FEATOMIC_IMPLIB_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    else()
        install(FILES ${FEATOMIC_SHARED_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    endif()
endif()

if (FEATOMIC_INSTALL_BOTH_STATIC_SHARED OR NOT BUILD_SHARED_LIBS)
    install(FILES ${FEATOMIC_STATIC_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
endif()

install(FILES
    ${PROJECT_BINARY_DIR}/featomic-config-version.cmake
    ${PROJECT_BINARY_DIR}/featomic-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/featomic
)
