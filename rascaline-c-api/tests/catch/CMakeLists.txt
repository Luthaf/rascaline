add_library(catch STATIC catch.cpp)
target_include_directories(catch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(catch PUBLIC cxx_std_17)
