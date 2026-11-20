add_executable(test_bootstrap test_bootstrap.cpp)
target_include_directories(test_bootstrap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_bootstrap PRIVATE engine CUDA::cudart)
add_test(NAME test_bootstrap COMMAND test_bootstrap)
set_tests_properties(test_bootstrap PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_loader test_loader.cpp)
target_include_directories(test_loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_loader PRIVATE engine CUDA::cudart)
add_test(NAME test_loader COMMAND test_loader)
set_tests_properties(test_loader PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_tokenizer test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_tokenizer PRIVATE engine CUDA::cudart)
add_test(NAME test_tokenizer COMMAND test_tokenizer)
set_tests_properties(test_tokenizer PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_cpu_ops test_cpu_ops.cpp)
target_include_directories(test_cpu_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_cpu_ops PRIVATE engine CUDA::cudart)
add_test(NAME test_cpu_ops COMMAND test_cpu_ops)
set_tests_properties(test_cpu_ops PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_cuda_ops test_cuda_ops.cpp)
target_include_directories(test_cuda_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_cuda_ops PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME test_cuda_ops COMMAND test_cuda_ops)
set_tests_properties(test_cuda_ops PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_kv_cache test_kv_cache.cpp)
target_include_directories(test_kv_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_kv_cache PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME test_kv_cache COMMAND test_kv_cache)
set_tests_properties(test_kv_cache PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_sampler test_sampler.cpp)
target_include_directories(test_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_sampler PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME test_sampler COMMAND test_sampler)
set_tests_properties(test_sampler PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(test_attention test_attention.cpp)
target_include_directories(test_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_attention PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME test_attention COMMAND test_attention)
set_tests_properties(test_attention PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")



add_executable(test_quant test_quant.cpp)
target_include_directories(test_quant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_quant PRIVATE engine CUDA::cudart CUDA::cublas)
add_test(NAME test_quant COMMAND test_quant)
set_tests_properties(test_quant PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(bench_launch_floor bench_launch_floor.cu)
target_include_directories(bench_launch_floor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(bench_launch_floor PRIVATE engine CUDA::cudart CUDA::cublas)
set_target_properties(bench_launch_floor PROPERTIES CUDA_ARCHITECTURES "89")

add_executable(check_fused_logits check_fused_logits.cpp)
target_include_directories(check_fused_logits PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(check_fused_logits PRIVATE engine CUDA::cudart CUDA::cublas)



add_executable(bench_falsifying_test bench_falsifying_test.cu)
target_link_libraries(bench_falsifying_test PRIVATE )

add_executable(bench_real_shapes bench_real_shapes.cu)
target_link_libraries(bench_real_shapes PRIVATE )

add_executable(check_splitk_vs_legacy check_splitk_vs_legacy.cpp)
target_include_directories(check_splitk_vs_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(check_splitk_vs_legacy PRIVATE engine CUDA::cudart CUDA::cublas)

add_executable(debug_step1 debug_step1.cpp)
target_include_directories(debug_step1 PRIVATE /..)
target_link_libraries(debug_step1 PRIVATE engine CUDA::cudart CUDA::cublas)

add_executable(check_teacher_forced_gate check_teacher_forced_gate.cpp)
target_include_directories(check_teacher_forced_gate PRIVATE /..)
target_link_libraries(check_teacher_forced_gate PRIVATE engine CUDA::cudart CUDA::cublas)

add_executable(check_ppl_splitk_vs_legacy check_ppl_splitk_vs_legacy.cpp)
target_include_directories(check_ppl_splitk_vs_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(check_ppl_splitk_vs_legacy PRIVATE engine CUDA::cudart CUDA::cublas)

add_executable(profile_single_kernel profile_single_kernel.cu)
target_link_libraries(profile_single_kernel PRIVATE CUDA::cudart)
set_target_properties(profile_single_kernel PROPERTIES CUDA_ARCHITECTURES "89")

add_executable(bench_wave_sweep bench_wave_sweep.cu)
target_link_libraries(bench_wave_sweep PRIVATE CUDA::cudart)
set_target_properties(bench_wave_sweep PROPERTIES CUDA_ARCHITECTURES "89")

add_executable(bench_wave_single bench_wave_single.cu)
target_link_libraries(bench_wave_single PRIVATE CUDA::cudart)
set_target_properties(bench_wave_single PROPERTIES CUDA_ARCHITECTURES "89")

add_executable(bench_persistent_shape bench_persistent_shape.cu)
target_include_directories(bench_persistent_shape PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(bench_persistent_shape PRIVATE CUDA::cudart)
set_target_properties(bench_persistent_shape PROPERTIES CUDA_ARCHITECTURES "89")
