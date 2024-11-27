include(CMakeFindDependencyMacro)

# use the same version for featomic as the main CMakeLists.txt
set(REQUIRED_FEATOMIC_VERSION @REQUIRED_FEATOMIC_VERSION@)
find_package(featomic ${REQUIRED_FEATOMIC_VERSION} CONFIG REQUIRED)

# use the same version for metatensor_torch as the main CMakeLists.txt
set(REQUIRED_METATENSOR_TORCH_VERSION @REQUIRED_METATENSOR_TORCH_VERSION@)
find_package(metatensor_torch ${REQUIRED_METATENSOR_TORCH_VERSION} CONFIG REQUIRED)

# We can only load metatensorfeatomic_torch with the same minor version of Torch
# that was used to compile it (and is stored in BUILD_TORCH_VERSION)
set(BUILD_TORCH_VERSION @Torch_VERSION@)
set(BUILD_TORCH_MAJOR @Torch_VERSION_MAJOR@)
set(BUILD_TORCH_MINOR @Torch_VERSION_MINOR@)

find_package(Torch ${BUILD_TORCH_VERSION} REQUIRED)

if (NOT "${BUILD_TORCH_MAJOR}" STREQUAL "${Torch_VERSION_MAJOR}")
    message(FATAL_ERROR "found incompatible torch version: featomic-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()

if (NOT "${BUILD_TORCH_MINOR}" STREQUAL "${Torch_VERSION_MINOR}")
    message(FATAL_ERROR "found incompatible torch version: featomic-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()


include(${CMAKE_CURRENT_LIST_DIR}/featomic_torch-targets.cmake)
