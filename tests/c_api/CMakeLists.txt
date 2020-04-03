cmake_minimum_required(VERSION 3.6)

project(rascaline-capi-tests CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
    message(STATUS "Setting build type to 'debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "debug" CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

set(CARGO_ROOT "${PROJECT_SOURCE_DIR}/../../")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_ARG "")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "--release")
else()
    message(FATAL_ERROR "unsuported CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES ${CARGO_ROOT}/src/**.rs)
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/librascaline.built
    COMMAND cargo build ${CARGO_BUILD_ARG}
    COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/librascaline.built
    WORKING_DIRECTORY ${CARGO_ROOT}
    DEPENDS ${ALL_RUST_SOURCES}
)
add_custom_target(ensure-rust-is-built DEPENDS ${PROJECT_BINARY_DIR}/librascaline.built)

add_library(rascaline SHARED IMPORTED)
target_include_directories(rascaline INTERFACE "${PROJECT_SOURCE_DIR}/../../include/")

set_target_properties(rascaline
    PROPERTIES IMPORTED_LOCATION
    "${CARGO_ROOT}/target/${CMAKE_BUILD_TYPE}/${CMAKE_SHARED_LIBRARY_PREFIX}rascaline${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

add_library(catch STATIC external/catch.cpp)
target_include_directories(catch PUBLIC external)

file(GLOB ALL_TESTS *.cpp)
enable_testing()
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    add_dependencies(${_name_} ensure-rust-is-built)
    target_link_libraries(${_name_} rascaline catch)
    add_test(${_name_} ${_name_})
endforeach()
