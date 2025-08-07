/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <torch/torch.h>
#include <torch/nativert/ModelRunnerHandle.h> 
#include <unordered_map>
#include <vector>

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

    auto x = torch::rand({2, 2});

    torch::nativert::ModelRunnerHandle model_runner(package_path, model_name);
    std::vector<c10::IValue> inputs;
    inputs.push_back(x);
    auto out = model_runner.runWithFlatInputsAndOutputs(inputs);
    std::cout << out[0].toTensor().const_data_ptr<float>()[0] << std::endl;
    return 0;


}
