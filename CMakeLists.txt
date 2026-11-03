cmake_minimum_required(VERSION 3.24)
project(llm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4060 Laptop (Ada)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE ENGINE_SRC CONFIGURE_DEPENDS
  src/common/*.cpp src/loader/*.cpp src/tokenizer/*.cpp src/cpu/*.cpp
  src/model/*.cpp src/model/*.cu src/runtime/*.cpp src/cuda/*.cu src/runtime/*.cu
)

add_library(engine STATIC ${ENGINE_SRC})
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} src third_party)
target_link_libraries(engine PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(generate apps/generate.cpp)
target_link_libraries(generate PRIVATE engine)

add_executable(bench_read_bandwidth apps/bench_read_bandwidth.cpp)
target_link_libraries(bench_read_bandwidth PRIVATE engine)

add_executable(bench_attention apps/bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE engine)

add_executable(profile_single_step apps/profile_single_step.cpp)
target_link_libraries(profile_single_step PRIVATE engine)

add_executable(tokenize_stdin apps/tokenize_stdin.cpp)
target_link_libraries(tokenize_stdin PRIVATE engine)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(engine_cpp src/bindings/module.cpp)
  target_link_libraries(engine_cpp PRIVATE engine)
endif()

enable_testing()
add_subdirectory(tests)

