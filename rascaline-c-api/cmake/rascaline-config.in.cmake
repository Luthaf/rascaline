@PACKAGE_INIT@

if(rascaline_FOUND)
    return()
endif()

cmake_minimum_required(VERSION 3.16)
enable_language(CXX)

if (@BUILD_SHARED_LIBS@)
    add_library(rascaline SHARED IMPORTED GLOBAL)
else()
    add_library(rascaline STATIC IMPORTED GLOBAL)
endif()

# check that all expected files exist
if (NOT EXISTS ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_LIB_NAME@)
    message(FATAL_ERROR "unable to find rascaline library at ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_LIB_NAME@")
endif()

if (NOT EXISTS ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/rascaline.h)
    message(FATAL_ERROR "unable to find rascaline header at ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/rascaline.h")
endif()

if (NOT EXISTS ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/rascaline.hpp)
    message(FATAL_ERROR "unable to find rascaline header at ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/rascaline.hpp")
endif()


# create an imported rascaline target
set_target_properties(rascaline PROPERTIES
    IMPORTED_LOCATION ${PACKAGE_PREFIX_DIR}/@LIB_INSTALL_DIR@/@RASCALINE_LIB_NAME@
    INTERFACE_INCLUDE_DIRECTORIES ${PACKAGE_PREFIX_DIR}/@INCLUDE_INSTALL_DIR@/
    # we might need to link with a C++ compiler to get the C++ stdlib for
    # chemfiles, if the chemfiles feature is enabled
    IMPORTED_LINK_INTERFACE_LANGUAGES CXX
)


find_package(equistore @EQUISTORE_REQUIRED_VERSION@ REQUIRED CONFIG)
target_link_libraries(rascaline INTERFACE equistore)
target_compile_features(rascaline INTERFACE cxx_std_11)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT @BUILD_SHARED_LIBS@)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    # the rust standard lib uses pthread and libdl on linux
    target_link_libraries(rascaline INTERFACE Threads::Threads dl)
endif()
