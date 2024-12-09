cmake_minimum_required(VERSION 3.16)

project(featomic-test-cmake-project C CXX)

# We need to update the REQUIRED_FEATOMIC_VERSION in the same way we update the
# featomic version for dev builds
include(../../cmake/dev-versions.cmake)
set(REQUIRED_FEATOMIC_VERSION "0.6.0-rc2")
create_development_version("${REQUIRED_FEATOMIC_VERSION}" FEATOMIC_FULL_VERSION "featomic-v")
string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_FEATOMIC_VERSION ${FEATOMIC_FULL_VERSION})
find_package(featomic ${REQUIRED_FEATOMIC_VERSION} REQUIRED)

add_executable(c-main src/main.c)
target_link_libraries(c-main featomic)

add_executable(cxx-main src/main.cpp)
target_link_libraries(cxx-main featomic)
