add_executable(test_bootstrap test_bootstrap.cpp)
target_include_directories(test_bootstrap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_bootstrap PRIVATE engine CUDA::cudart)

add_test(NAME test_bootstrap COMMAND test_bootstrap)

