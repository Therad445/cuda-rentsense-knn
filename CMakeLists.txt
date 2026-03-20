cmake_minimum_required(VERSION 3.18)
project(cuda_rentsense_knn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_rentsense_knn
    src/main.cpp
    src/dataset.cpp
    src/knn_cpu.cpp
    src/knn_cuda.cu
)

target_include_directories(cuda_rentsense_knn PRIVATE include)
target_link_libraries(cuda_rentsense_knn PRIVATE CUDA::cudart)

target_compile_options(cuda_rentsense_knn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
)

set_target_properties(cuda_rentsense_knn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
