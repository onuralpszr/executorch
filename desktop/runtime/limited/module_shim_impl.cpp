/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <iostream>

namespace executorch {
namespace desktop {

// Here is where the aoti bouncers are going to be defined.
// I define the globals aoti generated compiled code calls
// They can be backed by ET systems

using namespace std;
using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

extern "C" {

using ModuleHandle = executorch::extension::Module*;
using AOTIRuntimeError = Error;
using AOTITorchError = Error;
using AtenTensorHandle = Tensor*;

AOTITorchError aoti_torch_create_tensor_from_blob(
  void* data,
  int64_t ndim,
  const int64_t* sizes_ptr,
  const int64_t* strides_ptr,
  int64_t storage_offset,
  int32_t dtype,
  int32_t device_type,
  int32_t device_index,
  AtenTensorHandle* ret_new_tensor);


AOTITorchError torch_load_module_from_file(char* package_path, int64_t package_path_len, char* model_name, int64_t model_name_len, ModuleHandle* ret_value) {
  (void)model_name;
  (void)model_name_len;
  *ret_value = new executorch::extension::Module(std::string(package_path, package_path_len));
  return Error::Ok;
}

AOTITorchError torch_delete_module_object(ModuleHandle handle) {
  delete handle;
  return Error::Ok;
}


AOTITorchError torch_module_forward_flattened(ModuleHandle handle, AtenTensorHandle arg, AtenTensorHandle* ret_value) {
  auto out = handle->forward(*arg);
  if (!out.ok()) {
    return out.error();
  }

  // Im sure this is a very stupid way of doing this, I will think of a better way later
  auto& t = out.get()[0].toTensor();

  // ETensor uses int32 for these things today
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  for (int i = 0; i < t.dim(); i++) {
    sizes.push_back(t.sizes()[i]);
    strides.push_back(t.strides()[i]);
  }
  AOTITorchError err = aoti_torch_create_tensor_from_blob(t.mutable_data_ptr(), t.dim(), sizes.data(), strides.data(), 0, (int32_t)t.dtype(), 0, 0, ret_value);
  (void)err;
  return Error::Ok;
}


}

} // namespace desktop
} // namespace executorch
