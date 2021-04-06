file(GLOB ALL_TESTS *.cpp)

foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(cxx-${_name_} ${_file_})
    target_link_libraries(cxx-${_name_} rascaline catch)
    add_test(
        NAME cxx-${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:cxx-${_name_}>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()
