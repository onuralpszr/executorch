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
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/executorch/shim/module_shim.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;
using executorch::runtime::EValue;
using executorch::extension::module::Module;

extern "C" {

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

} // extern "C"

namespace {
  AOTITorchError to_evalue(const TypedStableIValue& v, EValue* ret_val) {
    switch (v.tag) {
      case StableIValueTag::None:
        *ret_val = EValue();
        return 0;
      case StableIValueTag::Int:
        *ret_val = EValue(to<int64_t>(v.val));
        return 0;
      case StableIValueTag::Bool:
        *ret_val = EValue(to<bool>(v.val));
        return 0;
      case StableIValueTag::Double:
        *ret_val = EValue(to<double>(v.val));
        return 0;
      case StableIValueTag::Tensor:
        *ret_val = EValue(*reinterpret_cast<Tensor*>(to<AtenTensorHandle>(v.val)));
        return 0;
      default:
        return static_cast<AOTITorchError>(Error::InvalidArgument);
        
    }
  }

  AOTITorchError from_evalue(const EValue& v, TypedStableIValue* ret_val) {
    if (v.isNone()) {
      *ret_val = TypedStableIValue{from(std::nullopt), StableIValueTag::None};
      return 0;
    } else if (v.isInt()) {
      *ret_val = TypedStableIValue{from(v.toInt()), StableIValueTag::Int};
      return 0;
    } else if (v.isBool()) {
      *ret_val = TypedStableIValue{from(v.toBool()), StableIValueTag::Int};
      return 0;
    } else if (v.isDouble()) {
      *ret_val = TypedStableIValue{from(v.toDouble()), StableIValueTag::Int};
      return 0;
    } else if (v.isTensor()) {
      AtenTensorHandle ath = reinterpret_cast<AtenTensorHandle>(new Tensor(v.toTensor()));
      *ret_val = TypedStableIValue{from(ath), StableIValueTag::Tensor};
      return 0;
    } else {
      return static_cast<AOTITorchError>(Error::InvalidArgument);
    }
  }
} // namespace


AOTITorchError experimental_torch_load_module_from_file(const char* package_path, uint64_t package_path_len, const char* model_name, uint64_t model_name_len, ModuleHandle* ret_value) {
  (void)model_name;
  (void)model_name_len;
  *ret_value = reinterpret_cast<ModuleHandle>(new Module(std::string(package_path, package_path_len)));
  return 0;
}

AOTITorchError experimental_torch_delete_module_object(ModuleHandle handle) {
  delete reinterpret_cast<Module*>(handle);
  return 0;
}

AOTITorchError experimental_torch_module_num_outputs(ModuleHandle handle, uint64_t* ret_value) {
  auto meta = reinterpret_cast<Module*>(handle)->method_meta("forward");
  if (!meta.ok()) {
    return static_cast<AOTITorchError>(meta.error());
  }
  *ret_value = meta->num_outputs();
  return 0;
}


AOTITorchError experimental_torch_module_forward_flattened(ModuleHandle handle, const TypedStableIValue* args, uint64_t num_args, TypedStableIValue* ret_values, uint64_t num_outputs) {
  std::vector<EValue> vec;
  vec.reserve(num_args);
  for (uint64_t i = 0; i < num_args; ++i) {
    auto err = to_evalue(args[i], &vec.emplace_back());
    if (err != 0) {
      return err;
    }
  }

  auto res = reinterpret_cast<Module*>(handle)->forward(vec);
  if (!res.ok()) {
    return static_cast<AOTITorchError>(res.error());
  }

  std::vector<EValue>& out = res.get();
  if (out.size() != num_outputs) {
    return static_cast<AOTITorchError>(Error::InvalidArgument);
  }

  for (uint64_t i = 0; i < num_outputs; ++i) {
    auto err = from_evalue(out[i], &ret_values[i]);
    if (err != 0) {
      return err;
    }
  }
  return 0;
}
