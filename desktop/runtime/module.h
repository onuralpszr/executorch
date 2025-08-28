#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/executorch/shim/module_shim.h>
#include <memory>


namespace executorch::desktop {

class Module {
 private:
  std::shared_ptr<ModuleOpaque> m_;

 public:

  explicit Module(const std::string& package_path, const std::string& model_name) {
    ModuleHandle m = nullptr;
    auto err = experimental_torch_load_module_from_file(package_path.c_str(), package_path.size(), model_name.c_str(), model_name.size(), &m);
    if (err != 0) {
      throw std::runtime_error("Failed to load module");
    }
    m_ = std::shared_ptr<ModuleOpaque>(m, [](ModuleHandle m) {
      auto err = experimental_torch_delete_module_object(m);
      if (err != 0) {
        throw std::runtime_error("Failed to delete module");
      }
    });
  }

  // Copy and move constructors can be default because the underlying handle is a
  // shared_ptr
  Module(const Module& other) = default;
  Module(Module&& other) noexcept = default;

  // Copy and move assignment operators can be default because the underlying handle
  // is a shared_ptr
  Module& operator=(const Module& other) = default;
  Module& operator=(Module&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  ~Module() = default;

  std::vector<TypedStableIValue> forward_flattened(const std::vector<TypedStableIValue>& args) const {
    uint64_t num_outs = 0;
    auto err = experimental_torch_module_num_outputs(m_.get(), &num_outs);
    if (err != 0) {
      throw std::runtime_error("Failed to get number of outputs");
    }
    std::vector<TypedStableIValue> ret_value(num_outs);
    err = experimental_torch_module_forward_flattened(m_.get(), args.data(), args.size(), ret_value.data(), num_outs);
    if (err != 0) {
      throw std::runtime_error("Failed to run forward");
    }
    return ret_value;
  }

};

} // namespace torch::executorch::experimental
