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


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION RASCALINE_TORCH_VERSION)
string(STRIP ${RASCALINE_TORCH_VERSION} RASCALINE_TORCH_VERSION)

project(rascaline_torch
    VERSION ${RASCALINE_TORCH_VERSION}
    LANGUAGES CXX
)

option(RASCALINE_TORCH_TESTS "Build rascaline-torch C++ tests" OFF)
option(RASCALINE_TORCH_FETCH_METATENSOR_TORCH "Download and build the metatensor_torch library before building rascaline_torch" OFF)

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

set(REQUIRED_RASCALINE_VERSION "0.1")
find_package(rascaline ${REQUIRED_RASCALINE_VERSION} CONFIG REQUIRED)

# FindCUDNN.cmake distributed with PyTorch is a bit broken, so we have a
# fixed version in `cmake/FindCUDNN.cmake`
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(Torch 1.12 REQUIRED)

# ============================================================================ #
# Setup metatensor_torch

set(METATENSOR_FETCH_VERSION "0.5.5")
set(REQUIRED_METATENSOR_TORCH_VERSION "0.5")
if (RASCALINE_TORCH_FETCH_METATENSOR_TORCH)
    message(STATUS "Fetching metatensor-torch @ ${METATENSOR_FETCH_VERSION} from github")

    set(URL_ROOT "https://github.com/lab-cosmo/metatensor/releases/download")
    include(FetchContent)
    FetchContent_Declare(
        metatensor_torch
        URL      ${URL_ROOT}/metatensor-torch-v${METATENSOR_FETCH_VERSION}/metatensor-torch-cxx-${METATENSOR_FETCH_VERSION}.tar.gz
        URL_HASH SHA256=dac306ab59ac8b59167827405f468397dbf0d4a69988fce7b9f4285f2816a57c
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
endif()


set(RASCALINE_TORCH_HEADERS
    "include/rascaline/torch/system.hpp"
    "include/rascaline/torch/autograd.hpp"
    "include/rascaline/torch/calculator.hpp"
    "include/rascaline/torch.hpp"
)

set(RASCALINE_TORCH_SOURCE
    "src/system.cpp"
    "src/autograd.cpp"
    "src/openmp.cpp"
    "src/calculator.cpp"
    "src/register.cpp"
)

add_library(rascaline_torch SHARED
    ${RASCALINE_TORCH_HEADERS}
    ${RASCALINE_TORCH_SOURCE}
)

target_link_libraries(rascaline_torch PUBLIC torch metatensor_torch rascaline)
target_compile_features(rascaline_torch PUBLIC cxx_std_17)
target_include_directories(rascaline_torch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Create a header defining RASCALINE_TORCH_EXPORT for to export classes/functions
# in DLL on Windows.
set_target_properties(rascaline_torch PROPERTIES
    # hide non-exported symbols by default, this mimics Windows behavior on Unix
    CXX_VISIBILITY_PRESET hidden
)

if (RASCALINE_TORCH_FETCH_METATENSOR_TORCH)
    # If we install metatensor_torch together with rascaline_torch, we need to
    # set the RPATH to $ORIGIN to make sure rascaline_torch can find
    # metatensor_torch.
    set_target_properties(rascaline_torch PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

include(GenerateExportHeader)
generate_export_header(rascaline_torch
    BASE_NAME RASCALINE_TORCH
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/rascaline/torch/exports.h
)
target_compile_definitions(rascaline_torch PRIVATE rascaline_torch_EXPORTS)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    # Torch bundles its own copy of the OpenMP runtime library, and if we
    # compile and link against the system version as well this can lead to
    # crashes during initialization on macOS.
    #
    # So on this plaftorm we instead compile the code with OpenMP flags, and
    # leave the corresponding symbols undefined in `rascaline_torch`, hopping
    # that when Torch is loaded we'll get these symbols in the global namespace.
    #
    # On other platforms, this seems to be less of an issue, maybe because torch
    # adds a hash to the library name it bundles (i.e. `libgomp-de42aff.so`)
    if (APPLE)
        string(REPLACE " " ";" omp_cxx_flags_list ${OpenMP_CXX_FLAGS})
        target_compile_options(rascaline_torch PRIVATE ${omp_cxx_flags_list})
        target_include_directories(rascaline_torch PRIVATE SYSTEM ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(rascaline_torch PRIVATE -Wl,-undefined,dynamic_lookup)
    else()
        target_link_libraries(rascaline_torch PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if (RASCALINE_TORCH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    rascaline_torch-config-version.cmake
    VERSION ${RASCALINE_TORCH_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(TARGETS rascaline_torch
    EXPORT rascaline_torch-targets
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)
install(EXPORT rascaline_torch-targets
    DESTINATION ${LIB_INSTALL_DIR}/cmake/rascaline_torch
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rascaline_torch-config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rascaline_torch-config.cmake
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rascaline_torch-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rascaline_torch-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/rascaline_torch
)

install(DIRECTORY "include/rascaline" DESTINATION ${INCLUDE_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/rascaline DESTINATION ${INCLUDE_INSTALL_DIR})
