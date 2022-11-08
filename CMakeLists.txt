cmake_minimum_required(VERSION 3.18)
project(hpc_heat_diffusion CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(hpc_heat_diffusion src/cuda-shared.cu src/main.c src/cuda.cu)

set_target_properties(hpc_heat_diffusion PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
