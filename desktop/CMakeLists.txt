# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.29)

project(executorch_desktop)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC -Wall)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(_limited_runtime_sources runtime/limited/shim_tensor_impl.cpp runtime/limited/module_shim_impl.cpp)
add_library(limited_runtime STATIC ${_limited_runtime_sources})
target_include_directories(
  limited_runtime
  PUBLIC
    $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(limited_runtime PUBLIC -fexceptions -frtti -fPIC)
# Ensure symbols are exported properly
target_link_options(limited_runtime PUBLIC -Wl)
target_link_libraries(
  limited_runtime
  PUBLIC
    extension_tensor
    extension_module
)

find_package_torch_headers()

add_executable(executorch_desktop_limited runtime/runner.cpp)
target_compile_options(executorch_desktop_limited PUBLIC ${_common_compile_options})
target_include_directories(
  executorch_desktop_limited PUBLIC ${_common_include_directories}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(
  executorch_desktop_limited
  gflags
  limited_runtime
)

find_package_torch()

add_executable(executorch_desktop runtime/runner.cpp)
target_compile_options(executorch_desktop PUBLIC ${_common_compile_options})
target_include_directories(
  executorch_desktop PUBLIC ${_common_include_directories}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(
  executorch_desktop
  gflags
  torch
)
