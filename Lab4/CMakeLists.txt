
#set(JPEG_INCLUDE_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/include)
#set(JPEG_LIBRARY_DIR /share/apps/rc/software/libjpeg-turbo/2.1.4-GCCcore-12.2.0/lib)

#include_directories(${JPEG_INCLUDE_DIR})
#add_executable(lab3t lab3.cu)
#add_subdirectory(gpu/lab1)
add_executable(task01 task01.cu)
add_executable(task02 task02.cu)
#add_subdirectory(gpu/text_snippets)add_executable(task3 task3.cu)