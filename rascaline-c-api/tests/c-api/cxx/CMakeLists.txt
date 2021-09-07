file(GLOB ALL_TESTS *.cpp)

foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    set(_name_ cxx-${_name_})
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} rascaline catch)
    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

file(GLOB EXAMPLES ../../../examples/*.cpp)
foreach(_file_ ${EXAMPLES})
    get_filename_component(_name_ ${_file_} NAME_WE)
    set(_name_ example-cxx-${_name_})
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} rascaline)
endforeach()

add_test(
    NAME example-cxx-compute-soap
    COMMAND ${TEST_COMMAND} $<TARGET_FILE:example-cxx-compute-soap> ${XYZ_EXAMPLE_FILE}
)

add_test(
    NAME example-cxx-profiling
    COMMAND ${TEST_COMMAND} $<TARGET_FILE:example-cxx-profiling> ${XYZ_EXAMPLE_FILE}
)
