#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <executorch/desktop/runtime/module_shim.h>
#include <climits>
#include <memory>

namespace torch::stable {

class Module {
 private:
  std::shared_ptr<ModuleOpaque> m_;

 public:

  // Construct a stable::Tensor from an AtenTensorHandle (ATH)
  // Steals ownership from the ATH
  explicit Module(ModuleHandle m)
      : m_(m, [](ModuleHandle m) {
          TORCH_ERROR_CODE_CHECK(torch_delete_module_object(m));
        }) {}

  // Copy and move constructors can be default cuz the underlying handle is a
  // shared_ptr
  Module(const Module& other) = default;
  Module(Module&& other) noexcept = default;

  // Copy and move assignment operators can be default cuz the underlying handle
  // is a shared_ptr
  Module& operator=(const Module& other) = default;
  Module& operator=(Module&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  ~Module() = default;

  // Returns a borrowed reference to the ModuleHandle
  ModuleHandle get() const {
    return m_.get();
  }

  torch::stable::Tensor forward_flattened(torch::stable::Tensor arg) const {
    AtenTensorHandle out;
    AtenTensorHandle arg_ath = arg.get();
    TORCH_ERROR_CODE_CHECK(torch_module_forward_flattened(m_.get(), arg_ath, &out));
    return torch::stable::Tensor(out);
  }

};

} // namespace torch::stable
