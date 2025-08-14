#ifdef __cplusplus
extern "C" {
#endif

struct ModuleOpaque;
using ModuleHandle = ModuleOpaque*;

AOTI_TORCH_EXPORT AOTITorchError torch_load_module_from_file(const char* package_path, int64_t package_path_len, const char* model_name, int64_t model_name_len, ModuleHandle* ret_value);
AOTI_TORCH_EXPORT AOTITorchError torch_delete_module_object(ModuleHandle handle);
AOTI_TORCH_EXPORT AOTITorchError torch_module_forward_flattened(ModuleHandle handle, AtenTensorHandle arg, AtenTensorHandle* ret_value);

#ifdef __cplusplus
} // extern "C"
#endif
