/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch {
namespace desktop {

// Here is where the aoti bouncers are going to be defined.
// I define the globals aoti generated compiled code calls
// They can be backed by ET systems

using namespace std;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::Tensor;
using executorch::runtime::Span;

extern "C" {
  
using AtenTensorHandle = Tensor*;

// TODO: We should get a proper one
struct CUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

using AOTIRuntimeError = Error;
using AOTITorchError = Error;

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
using AOTInductorStreamHandle = void*;
using AOTIProxyExecutorHandle = void*;

using AOTInductorModelContainerCreateWithDeviceFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

using AOTInductorModelContainerDeleteFunc =
    AOTIRuntimeError (*)(AOTInductorModelContainerHandle container_handle);

using AOTInductorModelContainerGetNumInputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerGetNumOutputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerRunFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

AOTInductorModelContainerCreateWithDeviceFunc
    AOTInductorModelContainerCreateWithDevice = nullptr;
AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete = nullptr;
AOTInductorModelContainerGetNumInputsFunc
    AOTInductorModelContainerGetNumInputs = nullptr;
AOTInductorModelContainerGetNumOutputsFunc
    AOTInductorModelContainerGetNumOutputs = nullptr;
AOTInductorModelContainerRunFunc AOTInductorModelContainerRun = nullptr;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;
std::unordered_set<std::shared_ptr<Tensor>> tensors;

int32_t aoti_torch_grad_mode_is_enabled() {
  // No autograd ever
  return false;
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  if (enabled) {
    throw std::runtime_error("Cannot enable autograd");
  }
}

AOTITorchError aoti_torch_get_dim(
    AtenTensorHandle tensor,
    int64_t* ret_dim) {
  *ret_dim = tensor->dim();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
  *ret_data_ptr = tensor->mutable_data_ptr();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
  // Storage offset is always 0 in ET
  *ret_storage_offset = 0;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
  auto it = tensor_to_strides.find(tensor);
  if (it == tensor_to_strides.end()) {
    std::vector<int64_t> strides(tensor->dim());
    auto tensor_strides = tensor->strides();
    for (int i = 0; i < tensor->dim(); i++) {
      strides[i] = tensor_strides[i];
    }
    it = tensor_to_strides.emplace(tensor, std::move(strides)).first;
  }
  *ret_strides = it->second.data();
  std::cout << "getting strides from tensor " << tensor << " with dim "
            << tensor->dim() << std::endl;
  for (int i = 0; i < tensor->dim(); i++) {
    std::cout << "strides " << i << " = " << *ret_strides[i] << std::endl;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor,
    int32_t* ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->scalar_type());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes) {
  auto it = tensor_to_sizes.find(tensor);
  if (it == tensor_to_sizes.end()) {
    std::vector<int64_t> sizes(tensor->dim());
    auto tensor_sizes = tensor->sizes();
    for (int i = 0; i < tensor->dim(); i++) {
      sizes[i] = tensor_sizes[i];
    }
    it = tensor_to_sizes.emplace(tensor, std::move(sizes)).first;
  }
  *ret_sizes = it->second.data();
  std::cout << "getting sizes from tensor " << tensor << " with dim "
            << tensor->dim() << std::endl;
  for (int i = 0; i < tensor->dim(); i++) {
    std::cout << "size " << i << " = " << *ret_sizes[i] << std::endl;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
  throw std::runtime_error("Cannot get storage size on ETensor");
}

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  throw std::runtime_error("Not creating Tensor from blob here");
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  std::cout << "Entering stream guard for device " << device_index << std::endl;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  std::cout << "Exiting stream guard" << std::endl;
  return Error::Ok;
}

int aoti_torch_device_type_cpu() {
  // Let's say cpu is 0 for ET as well
  return 0;
}

__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_cuda() {
  // Let's say cuda is 1 for ET as well
  return 1;
}

__attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_float32() {
  // Let assume the dtype here is all we will support
  return 6;
}

AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  std::cout << "Deleting " << tensor << " in the limited runtime" << std::endl;
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      tensors.erase(it);
      break; // Exit the loop once the element is found and removed
    }
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  // This requires us to reserve CUDA memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  if (dtype != 6) { // throw if not float32
    throw std::runtime_error("Need to implement empty_strided for non-float32");
  }

  int64_t nbytes = numel * 4;

  if (device_type == 0) { // cpu
    // std::cout << "Allocating " << nbytes << " bytes on CPU " << std::endl;
    ptr = malloc(nbytes);
    if (ptr == nullptr) {
      throw std::runtime_error("Failed to call malloc");
    }
  } else {
    throw std::runtime_error(
        "Need to implement empty_strided for non-CUDA non-CPU");
  }
  // std::cout << "Allocated " << nbytes << " bytes at " << ptr << ", sizes_ptr "
  //           << sizes_ptr << std::endl;

  // ETensor sizes
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = sizes_ptr[i];
  }
  // ETensor creation
  auto tensor = executorch::extension::make_tensor_ptr(sizes, ptr);

  // Store the tensor
  tensors.insert(tensor);

  // std::cout << "sizes.data(): " << sizes.data()
  //           << ", tensor->sizes().data(): " << tensor->sizes().data()
  //           << std::endl;
  // std::cout << "Size[0] of tensor " << tensor.get() << " is "
  //           << tensor->sizes()[0] << std::endl;
  *ret_new_tensor = tensor.get();
  return Error::Ok;
}

AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  aoti_torch_empty_strided(
      ndim,
      sizes_ptr,
      strides_ptr,
      dtype,
      device_type,
      device_index,
      ret_new_tensor);
  (*ret_new_tensor)->set_data(data);
  return Error::Ok;
}

}

} // namespace desktop
} // namespace executorch
