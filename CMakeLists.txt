cmake_minimum_required(VERSION 3.5)

project(audio_visual_synthesis)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(TensorFlow REQUIRED)

add_executable(audio_visual_synthesis main.cpp avs.cpp)

target_include_directories(audio_visual_synthesis PUBLIC ${OpenCV_INCLUDE_DIRS} ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(audio_visual_synthesis ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})