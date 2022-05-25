cmake_minimum_required(VERSION 3.10)

message(STATUS "Running CMake version ${CMAKE_VERSION}")

project(rascaline-capi-tests C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

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

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/rascaline)

add_library(catch STATIC catch/catch.cpp)
target_include_directories(catch PUBLIC catch)

add_library(tests_helpers STATIC helpers.cpp)
target_link_libraries(tests_helpers rascaline)


find_program(VALGRIND valgrind)
if (VALGRIND)
    message(STATUS "Running tests using valgrind")
    set(TEST_COMMAND
        "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
        "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
        "--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/valgrind.supp" "--gen-suppressions=all"
    )
else()
    set(TEST_COMMAND "")
endif()

file(GLOB ALL_TESTS *.cpp)
list(REMOVE_ITEM ALL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp")

enable_testing()
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} rascaline catch tests_helpers)
    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

# make sure example compile and run
file(GLOB EXAMPLES ../../examples/*.c)
foreach(_file_ ${EXAMPLES})
    get_filename_component(_name_ ${_file_} NAME_WE)
    set(_name_ example-${_name_})
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} rascaline)
endforeach()

set(XYZ_EXAMPLE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../../rascaline/examples/data/water.xyz)
add_test(
    NAME example-c-compute-soap
    COMMAND ${TEST_COMMAND} $<TARGET_FILE:example-compute-soap> ${XYZ_EXAMPLE_FILE}
)

add_test(
    NAME example-c-profiling
    COMMAND ${TEST_COMMAND} $<TARGET_FILE:example-profiling> ${XYZ_EXAMPLE_FILE}
)

# TODO: remove or re-enable
# add_subdirectory(cxx)
