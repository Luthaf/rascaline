cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # Timestamp for FetchContent
endif()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # use variables to set OPTIONS
endif()


if(NOT "${LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    set(LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
        message(STATUS "Running CMake version ${CMAKE_VERSION}")
    endif()
endif()


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION FEATOMIC_TORCH_VERSION)
string(STRIP ${FEATOMIC_TORCH_VERSION} FEATOMIC_TORCH_VERSION)

include(cmake/dev-versions.cmake)
create_development_version("${FEATOMIC_TORCH_VERSION}" FEATOMIC_TORCH_FULL_VERSION)
message(STATUS "Building featomic-torch v${FEATOMIC_TORCH_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" FEATOMIC_TORCH_VERSION ${FEATOMIC_TORCH_FULL_VERSION})
project(featomic_torch
    VERSION ${FEATOMIC_TORCH_VERSION}
    LANGUAGES CXX
)
set(PROJECT_VERSION ${FEATOMIC_TORCH_FULL_VERSION})


option(FEATOMIC_TORCH_TESTS "Build featomic-torch C++ tests" OFF)
option(FEATOMIC_TORCH_FETCH_METATENSOR_TORCH "Download and build the metatensor_torch library before building featomic_torch" OFF)

set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

# Set a default build type if none was specified
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(
            CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: none(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) debug release relwithdebinfo minsizerel."
            FORCE
        )
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug relwithdebinfo minsizerel none)
    endif()
endif()

function(check_compatible_versions _dependency_ _actual_ _requested_)
    if(${_actual_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_actual_major_ "${CMAKE_MATCH_1}")
        set(_actual_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse actual version: ${_actual_}")
    endif()

    if(${_requested_} MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_requested_major_ "${CMAKE_MATCH_1}")
        set(_requested_minor_ "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "Failed to parse requested version: ${_requested_}")
    endif()

    if (${_requested_major_} EQUAL 0 AND ${_actual_minor_} EQUAL ${_requested_minor_})
        # major version is 0 and same minor version, everything is fine
    elseif (${_actual_major_} EQUAL ${_requested_major_})
        # same major version, everything is fine
    else()
        # not compatible
        message(FATAL_ERROR "Incompatible versions for ${_dependency_}: we need v${_requested_}, but we got v${_actual_}")
    endif()
endfunction()


set(REQUIRED_FEATOMIC_VERSION "0.0.0")
if (NOT "$ENV{FEATOMIC_NO_LOCAL_DEPS}" STREQUAL "1")
    # If building a dev version, we also need to update the
    # REQUIRED_FEATOMIC_VERSION in the same way we update the
    # featomic-torch version
    create_development_version("${REQUIRED_FEATOMIC_VERSION}" FEATOMIC_FULL_VERSION)
else()
    set(FEATOMIC_FULL_VERSION ${REQUIRED_FEATOMIC_VERSION})
endif()
string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_FEATOMIC_VERSION ${FEATOMIC_FULL_VERSION})

# Either featomic is built as part of the same CMake project, or we try to
# find the corresponding CMake package
if (TARGET featomic)
    get_target_property(FEATOMIC_BUILD_VERSION featomic BUILD_VERSION)
    check_compatible_versions("featomic" ${FEATOMIC_BUILD_VERSION} ${REQUIRED_FEATOMIC_VERSION})
else()
    find_package(featomic ${REQUIRED_FEATOMIC_VERSION} CONFIG REQUIRED)

    get_target_property(FEATOMIC_LOCATION featomic IMPORTED_LOCATION)
    get_filename_component(FEATOMIC_LOCATION ${FEATOMIC_LOCATION} DIRECTORY)
    message(STATUS "Using local featomic from ${FEATOMIC_LOCATION}")
endif()


# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 1.12 REQUIRED)

# ============================================================================ #
# Setup metatensor_torch

# METATENSOR_FETCH_VERSION is the exact version we will fetch from github if
# FEATOMIC_TORCH_FETCH_METATENSOR_TORCH=ON, and REQUIRED_METATENSOR_TORCH_VERSION
# is the minimal version we require when using `find_package` to find the library.
#
# When updating METATENSOR_FETCH_VERSION, you will also have to update the
# SHA256 sum of the file in `FetchContent_Declare`.
set(METATENSOR_FETCH_VERSION "0.6.1")
set(REQUIRED_METATENSOR_TORCH_VERSION "0.6")
if (FEATOMIC_TORCH_FETCH_METATENSOR_TORCH)
    message(STATUS "Fetching metatensor-torch from github")

    set(URL_ROOT "https://github.com/lab-cosmo/metatensor/releases/download")
    include(FetchContent)
    FetchContent_Declare(
        metatensor_torch
        URL      ${URL_ROOT}/metatensor-torch-v${METATENSOR_FETCH_VERSION}/metatensor-torch-cxx-${METATENSOR_FETCH_VERSION}.tar.gz
        URL_HASH SHA256=0941da4bc6d25ee73b597774d3c8c6edf6a44f134139bd93b33834eae52ac4dd
    )

    if (CMAKE_VERSION VERSION_GREATER 3.18)
        FetchContent_MakeAvailable(metatensor_torch)
    else()
        if (NOT metatensor_POPULATED)
            FetchContent_Populate(metatensor_torch)
        endif()

        add_subdirectory(${metatensor_torch_SOURCE_DIR} ${metatensor_torch_BINARY_DIR})
    endif()
else()
    find_package(metatensor_torch ${REQUIRED_METATENSOR_TORCH_VERSION} REQUIRED CONFIG)
    get_target_property(METATENSOR_TORCH_LOCATION metatensor_torch IMPORTED_LOCATION)
    get_filename_component(METATENSOR_TORCH_LOCATION ${METATENSOR_TORCH_LOCATION} DIRECTORY)
    message(STATUS "Using local metatensor-torch from ${METATENSOR_TORCH_LOCATION}")
endif()


set(FEATOMIC_TORCH_HEADERS
    "include/featomic/torch/system.hpp"
    "include/featomic/torch/autograd.hpp"
    "include/featomic/torch/calculator.hpp"
    "include/featomic/torch.hpp"
)

set(FEATOMIC_TORCH_SOURCE
    "src/system.cpp"
    "src/autograd.cpp"
    "src/openmp.cpp"
    "src/calculator.cpp"
    "src/register.cpp"
)

add_library(featomic_torch SHARED
    ${FEATOMIC_TORCH_HEADERS}
    ${FEATOMIC_TORCH_SOURCE}
)

target_link_libraries(featomic_torch PUBLIC torch metatensor_torch featomic)
target_compile_features(featomic_torch PUBLIC cxx_std_17)
target_include_directories(featomic_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Create a header defining FEATOMIC_TORCH_EXPORT for to export classes/functions
# in DLL on Windows.
set_target_properties(featomic_torch PROPERTIES
    # hide non-exported symbols by default, this mimics Windows behavior on Unix
    CXX_VISIBILITY_PRESET hidden
)

if (FEATOMIC_TORCH_FETCH_METATENSOR_TORCH)
    # If we install metatensor_torch together with featomic_torch, we need to
    # set the RPATH to $ORIGIN to make sure featomic_torch can find
    # metatensor_torch.
    set_target_properties(featomic_torch PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

include(GenerateExportHeader)
generate_export_header(featomic_torch
    BASE_NAME FEATOMIC_TORCH
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/featomic/torch/exports.h
)
target_compile_definitions(featomic_torch PRIVATE featomic_torch_EXPORTS)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    # Torch bundles its own copy of the OpenMP runtime library, and if we
    # compile and link against the system version as well this can lead to
    # crashes during initialization on macOS.
    #
    # So on this plaftorm we instead compile the code with OpenMP flags, and
    # leave the corresponding symbols undefined in `featomic_torch`, hopping
    # that when Torch is loaded we'll get these symbols in the global namespace.
    #
    # On other platforms, this seems to be less of an issue, maybe because torch
    # adds a hash to the library name it bundles (i.e. `libgomp-de42aff.so`)
    if (APPLE)
        string(REPLACE " " ";" omp_cxx_flags_list ${OpenMP_CXX_FLAGS})
        target_compile_options(featomic_torch PRIVATE ${omp_cxx_flags_list})
        target_include_directories(featomic_torch PRIVATE SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(featomic_torch PRIVATE -Wl,-undefined,dynamic_lookup)
    else()
        target_link_libraries(featomic_torch PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if (FEATOMIC_TORCH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    featomic_torch-config-version.cmake
    VERSION ${FEATOMIC_TORCH_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS featomic_torch
    EXPORT featomic_torch-targets
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)
install(EXPORT featomic_torch-targets
    DESTINATION ${LIB_INSTALL_DIR}/cmake/featomic_torch
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/featomic_torch-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/featomic_torch-config.cmake
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/featomic_torch-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/featomic_torch-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/featomic_torch
)

install(DIRECTORY "include/featomic" DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/featomic DESTINATION ${INCLUDE_INSTALL_DIR})
