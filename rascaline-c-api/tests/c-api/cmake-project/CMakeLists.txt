cmake_minimum_required(VERSION 3.10)

project(rascaline-test-cmake-project C)

find_package(rascaline 0.1 REQUIRED)

add_executable(c-main src/main.c)

target_link_libraries(c-main rascaline)
