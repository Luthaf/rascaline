cmake_minimum_required(VERSION 3.10)

project(rascaline-test-cmake-project C CXX)

find_package(rascaline 0.1 REQUIRED)

add_executable(c-main src/main.c)
target_link_libraries(c-main rascaline)

if (${CMAKE_VERSION} VERSION_LESS 3.11)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

add_executable(cxx-main src/main.cpp)
target_link_libraries(cxx-main rascaline)
