cmake_minimum_required(VERSION 3.0)
project(rotatedyolo)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/cuda-10.2/include /home/ubuntu/wjd/json/single_include/)
link_directories(/usr/local/cuda-10.2/lib64)

set(SRC src/main.cpp
        src/rotated.cpp        
)
add_executable(a ${SRC})

target_link_libraries(a nvinfer cudart ${OpenCV_LIBS})

