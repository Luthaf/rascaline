# Basic CMake integration for rascaline.
cmake_minimum_required(VERSION 3.16)

# Is rascaline the main project configured by the user? Or is this being used
# as a submodule/subdirectory?
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(RASCALINE_MAIN_PROJECT ON)
else()
    set(RASCALINE_MAIN_PROJECT OFF)
endif()

if(${RASCALINE_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # Timestamp for FetchContent
endif()

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

# We follow the standard CMake convention of using BUILD_SHARED_LIBS to provide
# either a shared or static library as a default target. But since cargo always
# builds both versions by default, we also install both versions by default.
# `RASCALINE_INSTALL_BOTH_STATIC_SHARED=OFF` allow to disable this behavior, and
# only install the file corresponding to `BUILD_SHARED_LIBS=ON/OFF`.
#
# BUILD_SHARED_LIBS controls the `rascaline` cmake target, making it an alias of
# either `rascaline::static` or `rascaline::shared`. This is mainly relevant
# when using rascaline from another cmake project, either as a submodule or from
# an installed library (see cmake/rascaline-config.cmake)
option(BUILD_SHARED_LIBS "Use a shared library by default instead of a static one" ON)
option(RASCALINE_INSTALL_BOTH_STATIC_SHARED "Install both shared and static libraries" ON)

set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")
set(EXTRA_RUST_FLAGS "" CACHE STRING "Flags used to build rust code")
mark_as_advanced(RUST_BUILD_TARGET EXTRA_RUST_FLAGS)

option(RASCALINE_ENABLE_CHEMFILES "Disable the usage of chemfiles for reading structures from files" ON)
option(RASCALINE_FETCH_EQUISTORE "Download and build the equistore C API before building rascaline" OFF)

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")

if (${RASCALINE_MAIN_PROJECT})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if(${RASCALINE_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_BUILD_TYPE}" STREQUAL ${CMAKE_BUILD_TYPE})
    set(CACHED_LAST_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Last build type used in configuration")
    message(STATUS "Building rascaline in ${CMAKE_BUILD_TYPE} mode")
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

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target-dir=${CARGO_TARGET_DIR}")
# Handle cross compilation with RUST_BUILD_TARGET
if ("${RUST_BUILD_TARGET}" STREQUAL "")
    if (CARGO_VERSION_RAW MATCHES "host: ([a-zA-Z0-9_\\-]*)\n")
        set(RUST_BUILD_TARGET "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "failed to determine host target, output was: ${CARGO_VERSION_RAW}")
    endif()
endif()

set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")

if (NOT ${RASCALINE_ENABLE_CHEMFILES})
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--no-default-features")
endif()

# Get the list of libraries linked by default by cargo/rustc to add when linking
# to rascaline::static
if (CARGO_VERSION_CHANGED)
    # Adapted from https://github.com/corrosion-rs/corrosion/blob/dc1e4e5/cmake/FindRust.cmake
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/_cargo_required_libs")
    execute_process(
        COMMAND "${CARGO_EXE}" new --lib _cargo_required_libs
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE cargo_new_result
        ERROR_QUIET
    )

    if (cargo_new_result)
        message(FATAL_ERRPR "could not create empty project to find default static libs: ${cargo_new_result}")
    endif()

    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/_cargo_required_libs/Cargo.toml"
        "[workspace]\nmembers=[]\n[lib]\ncrate-type=[\"staticlib\"]")
    execute_process(
        COMMAND ${CARGO_EXE} rustc --verbose --color never --target=${RUST_BUILD_TARGET} -- --print=native-static-libs
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_cargo_required_libs"
        RESULT_VARIABLE cargo_build_result
        ERROR_VARIABLE cargo_build_error_message
    )

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

        set(CARGO_DEFAULT_LIBRARIES "${stripped_lib_list}" CACHE INTERNAL "list of implicitly linked libraries")
        message(STATUS "Cargo default link libraries are: ${CARGO_DEFAULT_LIBRARIES}")
    else()
        message(FATAL_ERROR "could not find default static libs: `native-static-libs` not found in: `${cargo_build_error_message}`")
    endif()
endif()

# ============================================================================ #
# Setup equistore

set(EQUISTORE_GIT_VERSION "c022fde")
set(EQUISTORE_REQUIRED_VERSION "0.1")
if (RASCALINE_FETCH_EQUISTORE)
    message(STATUS "Fetching equistore @ ${EQUISTORE_GIT_VERSION} from github")

    include(FetchContent)
    FetchContent_Declare(
        equistore
        URL      https://github.com/lab-cosmo/equistore/archive/${EQUISTORE_GIT_VERSION}.zip
        URL_HASH MD5=ff9f924ee7c24f2b237a60bec08e7270
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
else()
    find_package(equistore ${EQUISTORE_REQUIRED_VERSION} REQUIRED CONFIG)
endif()

# ============================================================================ #
# Setup rascaline libraries

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/../Cargo.toml
    ${PROJECT_SOURCE_DIR}/../rascaline/Cargo.toml
    ${PROJECT_SOURCE_DIR}/../rascaline/src/**.rs

    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/build.rs
    ${PROJECT_SOURCE_DIR}/src/**.rs
)

add_library(rascaline::shared SHARED IMPORTED GLOBAL)
set(RASCALINE_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rascaline${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(RASCALINE_IMPLIB_LOCATION "${RASCALINE_SHARED_LOCATION}.lib")

add_library(rascaline::static STATIC IMPORTED GLOBAL)
set(RASCALINE_STATIC_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rascaline${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(RASCALINE_SHARED_LIB_NAME ${RASCALINE_SHARED_LOCATION} NAME)
get_filename_component(RASCALINE_IMPLIB_NAME     ${RASCALINE_IMPLIB_LOCATION} NAME)
get_filename_component(RASCALINE_STATIC_LIB_NAME ${RASCALINE_STATIC_LOCATION} NAME)

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/libequistore.dylib`
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${RASCALINE_SHARED_LIB_NAME}")
    else() # LINUX
        # set the SONAME to libequistore.so
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-soname,${RASCALINE_SHARED_LIB_NAME}")
    endif()
else()
    set(CARGO_RUSTC_ARGS "")
endif()

if (NOT "${EXTRA_RUST_FLAGS}" STREQUAL "")
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};${EXTRA_RUST_FLAGS}")
endif()

add_custom_target(cargo-build-rascaline ALL
    COMMAND
        ${CMAKE_COMMAND} -E env
        MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        RUSTC_WRAPPER=$ENV{RUSTC_WRAPPER}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${RASCALINE_SHARED_LIB_NAME} and ${RASCALINE_STATIC_LIB_NAME} with cargo"
    BYPRODUCTS ${RASCALINE_STATIC_LOCATION} ${RASCALINE_SHARED_LOCATION} ${RASCALINE_IMPLIB_LOCATION}
)

add_dependencies(rascaline::shared cargo-build-rascaline)
add_dependencies(rascaline::static cargo-build-rascaline)

set(RASCALINE_HEADERS
    "${PROJECT_SOURCE_DIR}/include/rascaline.h"
    "${PROJECT_SOURCE_DIR}/include/rascaline.hpp"
)
set(RASCALINE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(rascaline::shared PROPERTIES
    IMPORTED_LOCATION ${RASCALINE_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE_DIR}
    # the library will need to be linked as C++ code
    # since it might contains chemfiles
    IMPORTED_LINK_INTERFACE_LANGUAGES CXX
)
target_compile_features(rascaline::shared INTERFACE cxx_std_11)

if (WIN32)
    set_target_properties(rascaline::shared PROPERTIES
        IMPORTED_IMPLIB ${RASCALINE_IMPLIB_LOCATION}
    )
endif()

set_target_properties(rascaline::static PROPERTIES
    IMPORTED_LOCATION ${RASCALINE_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${RASCALINE_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "${CARGO_DEFAULT_LIBRARIES}"
    # the library will need to be linked as C++ code
    # since it might contains chemfiles
    IMPORTED_LINK_INTERFACE_LANGUAGES CXX
)
target_compile_features(equistore::static INTERFACE cxx_std_11)


if (BUILD_SHARED_LIBS)
    add_library(equistore ALIAS equistore::shared)
else()
    add_library(equistore ALIAS equistore::static)
endif()

add_dependencies(cargo-build-rascaline equistore)
target_link_libraries(rascaline::shared INTERFACE equistore::shared)
target_link_libraries(rascaline::static INTERFACE equistore::shared)

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

if (RASCALINE_INSTALL_BOTH_STATIC_SHARED OR BUILD_SHARED_LIBS)
    if (WIN32)
        # DLL files should go in <prefix>/bin
        install(FILES ${RASCALINE_SHARED_LOCATION} DESTINATION ${BIN_INSTALL_DIR})
        # .lib files should go in <prefix>/lib
        install(FILES ${RASCALINE_IMPLIB_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    else()
        install(FILES ${RASCALINE_SHARED_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    endif()
endif()

if (RASCALINE_INSTALL_BOTH_STATIC_SHARED OR NOT BUILD_SHARED_LIBS)
    install(FILES ${RASCALINE_STATIC_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
endif()

install(FILES
    ${PROJECT_BINARY_DIR}/rascaline-config-version.cmake
    ${PROJECT_BINARY_DIR}/rascaline-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/rascaline
)
