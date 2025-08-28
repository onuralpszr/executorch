/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/desktop/runtime/module.h>

#include <gflags/gflags.h>
#include <torch/csrc/stable/tensor.h>
#include <unordered_map>
#include <vector>
#include <torch/csrc/stable/stableivalue_conversions.h>
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

AtenTensorHandle get_tensor(void* data) {
    std::vector<int64_t> sizes = {1, 1};
    std::vector<int64_t> strides = {1, 1};
    int32_t dtype = 6;
    int32_t device_type = 0;
    int32_t device_index = 0;
    AtenTensorHandle tensor_ptr;
    aoti_torch_create_tensor_from_blob(data, 2, sizes.data(), strides.data(), 0, dtype, device_type, device_index, &tensor_ptr);
    return tensor_ptr;
}

int32_t main(int32_t argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    float data = 1.0f;
    Tensor x(get_tensor(&data));
    std::cout << "Input Tensor, dim: " << x.dim() << " data: " << ((float*)x.data_ptr())[0] << std::endl;

    torch::stable::Module m(FLAGS_package_path, FLAGS_model_name);

    std::vector<TypedStableIValue> args;
    args.push_back(TypedStableIValue(from(x.get()), StableIValueTag::Tensor)); // TODO make from(Stable::Tensor) work with ET tensor shim
    std::vector<TypedStableIValue> out = m.forward_flattened(args);
    std::cout << "Output Tensor, dim: " << out.dim() << " data: " << ((float*)out.data_ptr())[0] << std::endl;

    return 0;
}
