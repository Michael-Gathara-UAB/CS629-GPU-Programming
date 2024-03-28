cmake_minimum_required(VERSION 3.18)
project(lab3 LANGUAGES CXX CUDA)

# Optionally, set the C++ standard
set(CMAKE_CXX_STANDARD 11)
#add_compile_options(-G)

# Set CUDA architecture for all targets
set(CMAKE_CUDA_ARCHITECTURES "80")
#set(JPEG_INCLUDE_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/include)
#set(JPEG_LIBRARY_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/lib)

#include_directories(${JPEG_INCLUDE_DIR})
#add_executable(lab3t lab3.cu)
#add_subdirectory(gpu/lab1)
add_executable(task01 task01.cu)
add_executable(task02 task02.cu)
add_executable(chantel chantel.cu)
#add_subdirectory(gpu/text_snippets)add_executable(task3 task3.cu)