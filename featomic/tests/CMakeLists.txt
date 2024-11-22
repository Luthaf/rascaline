cmake_minimum_required(VERSION 3.16)

message(STATUS "Running CMake version ${CMAKE_VERSION}")

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # Timestamp for FetchContent
endif()

project(featomic-capi-tests C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

option(FEATOMIC_ENABLE_COVERAGE "Collect code coverage for C and C++ API" OFF)

if(FEATOMIC_ENABLE_COVERAGE)
    message(STATUS "Collecting code coverage")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
            message(FATAL_ERROR "clang-cl is not supported for code coverage collection")
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    else()
        message(FATAL_ERROR "only GNU compilers and Clang are supported for code coverage collection")
    endif()

    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
        if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
            message(FATAL_ERROR "clang-cl is not supported for code coverage collection")
        endif()

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    else()
        message(FATAL_ERROR "only GNU compilers and Clang are supported for code coverage collection")
    endif()

    # This is taken from the output of `cargo tarpaulin --engine=llvm --print-rust-flags`
    set(RUST_FLAGS "-Cdebuginfo=2 --cfg=tarpaulin -Cinstrument-coverage -Clink-dead-code" CACHE INTERNAL "" FORCE)
endif()

# make sure assert works in tests
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -UNDEBUG")

if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
    message(STATUS "Setting build type to 'release' as none was specified.")
    set(CMAKE_BUILD_TYPE "release" CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/featomic)


# We use chemfiles in examples to show how to create System
set(BUILD_SHARED_LIBS OFF)
include(FetchContent)
FetchContent_Declare(
    chemfiles
    URL      https://github.com/chemfiles/chemfiles/releases/download/0.10.4/chemfiles-0.10.4.tar.gz
    URL_HASH SHA256=b8232ddaae2953538274982838aa6c2df87d300f7e2f80e92c171581e06325ba
)

message(STATUS "Fetching chemfiles from github")
if (CMAKE_VERSION VERSION_GREATER 3.18)
    FetchContent_MakeAvailable(chemfiles)
else()
    if (NOT metatensor_POPULATED)
        FetchContent_Populate(chemfiles)
    endif()

    add_subdirectory(${chemfiles_SOURCE_DIR} ${chemfiles_BINARY_DIR})
endif()


# Add featomic and metatensor to the rpath of tests, so we can load the shared
# library directly from `target/{debug,release}/`
get_target_property(FEATOMIC_LOCATION featomic::shared IMPORTED_LOCATION)
get_filename_component(FEATOMIC_DIRECTORY "${FEATOMIC_LOCATION}" DIRECTORY)

get_target_property(METATENSOR_LOCATION metatensor::shared IMPORTED_LOCATION)
get_filename_component(METATENSOR_DIRECTORY "${METATENSOR_LOCATION}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${FEATOMIC_DIRECTORY};${METATENSOR_DIRECTORY}")

add_subdirectory(utils/catch)

add_library(tests_helpers STATIC utils/helpers.cpp)
target_include_directories(tests_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils/)
target_link_libraries(tests_helpers featomic::shared)


find_program(VALGRIND valgrind)
if (VALGRIND)
    if (NOT "$ENV{FEATOMIC_DISABLE_VALGRIND}" EQUAL "1")
        message(STATUS "Running tests using valgrind")
        set(TEST_COMMAND
            "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
            "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
            "--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/utils/valgrind.supp" "--gen-suppressions=all"
        )
    endif()
else()
    set(TEST_COMMAND "")
endif()

# on windows, shared libraries are found in the PATH. This plays a similar role
# to `CMAKE_BUILD_RPATH` above
string(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
set(WINDOWS_TEST_PATH "${PATH_STRING}\;${METATENSOR_DIRECTORY}\;${FEATOMIC_DIRECTORY}")

set(XYZ_EXAMPLE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../featomic/examples/data/water.xyz)

enable_testing()
add_subdirectory(c)
add_subdirectory(cxx)
