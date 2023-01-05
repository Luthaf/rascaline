@PACKAGE_INIT@

if(rascaline_FOUND)
    return()
endif()

cmake_minimum_required(VERSION 3.10)

enable_language(CXX)
find_package(equistore 0.1 REQUIRED CONFIG)

if (@BUILD_SHARED_LIBS@)
    add_library(rascaline SHARED IMPORTED GLOBAL)
else()
    add_library(rascaline STATIC IMPORTED GLOBAL)
endif()

set_target_properties(rascaline PROPERTIES
    IMPORTED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_LIB_NAME@
    INTERFACE_INCLUDE_DIRECTORIES ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/
    # we need to link with a C++ compiler to get the C++ stdlib for chemfiles
    IMPORTED_LINK_INTERFACE_LANGUAGES CXX
)
target_link_libraries(rascaline INTERFACE equistore)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.11)
    # we can not set compile features for imported targete before cmake 3.11
    # users will have to manually request C++11
    target_compile_features(rascaline INTERFACE cxx_std_11)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT @BUILD_SHARED_LIBS@)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    # the rust standard lib uses pthread and libdl on linux
    target_link_libraries(rascaline INTERFACE Threads::Threads dl)
endif()
