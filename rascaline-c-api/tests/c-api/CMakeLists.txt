cmake_minimum_required(VERSION 3.10)

message(STATUS "Running CMake version ${CMAKE_VERSION}")

project(rascaline-capi-tests CXX)
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

file(GLOB ALL_TESTS *.cpp)
list(REMOVE_ITEM ALL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/helpers.cpp")

enable_testing()
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} rascaline catch tests_helpers)
    add_test(${_name_} ${_name_})

    # TODO: add support for running tests using valgrind
endforeach()

add_subdirectory(cxx)
