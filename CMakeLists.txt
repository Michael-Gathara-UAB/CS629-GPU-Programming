cmake_minimum_required(VERSION 3.18)
project(lab3 LANGUAGES CXX CUDA)

# Optionally, set the C++ standard
set(CMAKE_CXX_STANDARD 11)
#add_compile_options(-G)

# Set CUDA architecture for all targets
set(CMAKE_CUDA_ARCHITECTURES "80")

add_subdirectory(Lab4)