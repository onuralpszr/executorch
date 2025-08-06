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
set(_common_compile_options -Wno-deprecated-declarations -fPIC -Wall -Werror)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package_torch()

add_executable(executorch_desktop runtime/full/main.cpp)
target_compile_options(executorch_desktop PUBLIC ${_common_compile_options})
target_include_directories(
  executorch_desktop PUBLIC ${_common_include_directories}
)
target_link_libraries(
  executorch_desktop
  gflags
  torch
)
