cmake_minimum_required(VERSION 3.20)

project(ldlt-benchmark LANGUAGES CXX)

option(TAU_BENCHMARKING "Enable benchamrking with TAU" ON)
#set(CMAKE_CXX_COMPILER tau_cxx.sh)
#if(TAU_BENCHMARKING)
#	set(CMAKE_CXX_COMPILER tau_cxx.sh)
#endif()

add_subdirectory(src)