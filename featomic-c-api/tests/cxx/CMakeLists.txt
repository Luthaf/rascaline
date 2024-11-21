file(GLOB ALL_TESTS *.cpp)

foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    set(_name_ cxx-${_name_})
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} featomic::shared catch)
    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(WIN32)
        set_tests_properties(${_name_} PROPERTIES ENVIRONMENT "PATH=${WINDOWS_TEST_PATH}")
    endif()
endforeach()

file(GLOB EXAMPLES ../../examples/*.cpp)
foreach(_file_ ${EXAMPLES})
    get_filename_component(_name_ ${_file_} NAME_WE)
    set(_name_ example-cxx-${_name_})
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} featomic::shared chemfiles)

    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}> ${XYZ_EXAMPLE_FILE}
    )

    if(WIN32)
        set_tests_properties(${_name_} PROPERTIES ENVIRONMENT "PATH=${WINDOWS_TEST_PATH}")
    endif()
endforeach()
