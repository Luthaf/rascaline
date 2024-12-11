cmake_minimum_required(VERSION 3.16)

project(featomic-torch-test-cmake-project CXX)


# We need to update the REQUIRED_FEATOMIC_VERSION in the same way we update the
# featomic version for dev builds
include(../../cmake/dev-versions.cmake)
set(REQUIRED_FEATOMIC_TORCH_VERSION "0.6.0-rc1")
create_development_version("${REQUIRED_FEATOMIC_TORCH_VERSION}" FEATOMIC_TORCH_FULL_VERSION "featomic-torch-v")
string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_FEATOMIC_TORCH_VERSION ${FEATOMIC_TORCH_FULL_VERSION})
find_package(featomic_torch ${REQUIRED_FEATOMIC_TORCH_VERSION} REQUIRED)

add_executable(torch-main src/main.cpp)
target_link_libraries(torch-main featomic_torch)

enable_testing()
add_test(NAME torch-main COMMAND torch-main)

if(WIN32)
    # We need to set the path to allow access to metatensor.dll
    STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
    set_tests_properties(torch-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:torch>\;$<TARGET_FILE_DIR:metatensor::shared>\;$<TARGET_FILE_DIR:metatensor_torch>"
    )
endif()
