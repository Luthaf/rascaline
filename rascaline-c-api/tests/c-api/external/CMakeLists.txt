function(unpack_library _name_)
    set(_archive_ ${CMAKE_CURRENT_SOURCE_DIR}/${_name_}.tar.gz)
    file(SHA256 ${_archive_} _shasum_)
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${_name_}/${_shasum_}")
        message(STATUS "Unpacking ${_name_} sources")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${_name_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf ${_archive_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E touch ${_name_}/${_shasum_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${_name_}/.git")
        # .git directories can get huge and increase the size of this repository
        message(FATAL_ERROR "${_name_} archive contains a .git directory, please remove it")
    endif()

    # make sure cmake re-runs whenever the file changes
    get_directory_property(_previous_depends_ CMAKE_CONFIGURE_DEPENDS)
    set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${_previous_depends_};${_name_}.tar.gz")
endfunction()

function(external_library _name_)
    unpack_library(${_name_})
    add_subdirectory(
        ${CMAKE_CURRENT_BINARY_DIR}/${_name_}
        ${CMAKE_CURRENT_BINARY_DIR}/${_name_}
        EXCLUDE_FROM_ALL
    )
endfunction()

set(CATCH_INSTALL_EXTRAS OFF)
set(CATCH_INSTALL_DOCS OFF)
external_library(catch-a9ed2c2)
mark_as_advanced(FORCE
    CATCH_DEVELOPMENT_BUILD
    CATCH_INSTALL_DOCS
    CATCH_INSTALL_EXTRAS
)
