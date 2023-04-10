cmake_minimum_required(VERSION 3.13)
project(lfxavsynthesis)

set(CMAKE_CXX_STANDARD 14)

find_package(TensorFlow REQUIRED)
target_link_libraries(lfxavsynthesis TensorFlow::TensorFlow)