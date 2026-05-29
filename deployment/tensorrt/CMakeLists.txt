cmake_minimum_required(VERSION 3.20)
project(ray_budget_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${TENSORRT_ROOT}/include
          $ENV{TENSORRT_ROOT}/include
          /usr/include
          /usr/local/include)

find_library(TENSORRT_NVINFER
    NAMES nvinfer
    HINTS ${TENSORRT_ROOT}/lib
          ${TENSORRT_ROOT}/lib64
          $ENV{TENSORRT_ROOT}/lib
          $ENV{TENSORRT_ROOT}/lib64
          /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib)

find_library(TENSORRT_ONNXPARSER
    NAMES nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib
          ${TENSORRT_ROOT}/lib64
          $ENV{TENSORRT_ROOT}/lib
          $ENV{TENSORRT_ROOT}/lib64
          /usr/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER OR NOT TENSORRT_ONNXPARSER)
    message(FATAL_ERROR "TensorRT headers/libs not found. Set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

add_executable(trt_builder trt_builder.cpp)
target_include_directories(trt_builder PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_builder PRIVATE
    CUDA::cudart
    ${TENSORRT_NVINFER}
    ${TENSORRT_ONNXPARSER})

add_library(ray_budget_inference STATIC inference_harness.cpp)
target_include_directories(ray_budget_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR})
target_link_libraries(ray_budget_inference PUBLIC
    CUDA::cudart
    ${TENSORRT_NVINFER})

add_library(ray_budget_allocation_converter STATIC allocation_converter.cu)
set_target_properties(ray_budget_allocation_converter PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ray_budget_allocation_converter PUBLIC CUDA::cudart)

add_library(ray_budget_fallback STATIC fallback_manager.cu)
set_target_properties(ray_budget_fallback PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(ray_budget_fallback PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ray_budget_fallback PUBLIC
    ray_budget_allocation_converter
    CUDA::cudart)

add_library(ray_budget_vulkan_interop STATIC
    vulkan_interop_bridge.cpp
    vulkan_dispatch_snippet.cpp)
target_include_directories(ray_budget_vulkan_interop PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS})
target_link_libraries(ray_budget_vulkan_interop PUBLIC
    ray_budget_inference
    ray_budget_allocation_converter
    ray_budget_fallback
    Vulkan::Vulkan
    CUDA::cudart)

add_library(ray_budget_profile STATIC profile_harness.cpp)
target_include_directories(ray_budget_profile PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(ray_budget_profile PUBLIC
    ray_budget_inference
    ray_budget_allocation_converter
    ray_budget_fallback
    Vulkan::Vulkan
    CUDA::cudart)
