/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <torch/csrc/stable/tensor.h>
#include <unordered_map>
#include <vector>
#include <iostream>

using torch::stable::Tensor;


DEFINE_string(
    package_path,
    "model.pt2",
    "Model serialized in pt2 format.");
    
DEFINE_string(
    model_name,
    "forward",
    "Model name.");

int32_t main(int32_t argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const char* package_path = FLAGS_package_path.c_str();
    const char* model_name = FLAGS_model_name.c_str();

    std::vector<int64_t> sizes = {1, 1};
    std::vector<int64_t> strides = {1, 1};
    int32_t dtype = 6;
    int32_t device_type = 0;
    int32_t device_index = 0;
    AtenTensorHandle tensor_ptr;
    aoti_torch_empty_strided(2, sizes.data(), strides.data(), dtype, device_type, device_index, &tensor_ptr);
    Tensor x(tensor_ptr);
    std::cout << x.dim() << std::endl;
    return 0;
}
