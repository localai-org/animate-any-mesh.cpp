#include "internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>

namespace aam::detail {

backend_handle::~backend_handle() {
    if (value != nullptr) {
        ggml_backend_free(value);
    }
}

namespace {

void load_backends(const std::filesystem::path & directory) {
    static std::mutex mutex;
    static std::set<std::string> loaded_directories;
    const std::string key = directory.empty() ? std::string{} : directory.lexically_normal().string();
    std::scoped_lock lock{mutex};
    if (!loaded_directories.insert(key).second) return;
    if (directory.empty()) {
        ggml_backend_load_all();
    } else {
        ggml_backend_load_all_from_path(directory.string().c_str());
    }
}

bool matches(device_kind wanted, ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
    switch (wanted) {
    case device_kind::automatic:
        return type == GGML_BACKEND_DEVICE_TYPE_GPU;
    case device_kind::cpu:
        return type == GGML_BACKEND_DEVICE_TYPE_CPU;
    case device_kind::vulkan: {
        const auto reg = ggml_backend_dev_backend_reg(device);
        const char * name = reg == nullptr ? nullptr : ggml_backend_reg_name(reg);
        return name != nullptr && std::string_view{name} == "Vulkan";
    }
    }
    return false;
}

} // namespace

result<std::unique_ptr<backend_handle>> make_backend(const runtime_options & options) {
    load_backends(options.backend_directory);

    ggml_backend_dev_t selected = nullptr;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (matches(options.device, candidate)) {
            selected = candidate;
            break;
        }
    }
    if (selected == nullptr && options.device == device_kind::automatic) {
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * candidate = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                selected = candidate;
                break;
            }
        }
    }
    if (selected == nullptr) {
        return std::unexpected(fail(error_code::backend_unavailable,
            "requested ggml backend is unavailable"));
    }

    auto result_handle = std::make_unique<backend_handle>();
    result_handle->value = ggml_backend_dev_init(selected, nullptr);
    if (result_handle->value == nullptr) {
        return std::unexpected(fail(error_code::backend_unavailable,
            "ggml failed to initialize the selected backend"));
    }

    const char * description = ggml_backend_dev_description(selected);
    result_handle->name = description == nullptr ? ggml_backend_dev_name(selected) : description;

    if (ggml_backend_dev_type(selected) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        std::uint32_t threads = options.threads;
        if (threads == 0) {
            threads = std::max(1U, std::thread::hardware_concurrency());
        }
        const auto bounded = std::min<std::uint32_t>(threads,
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()));
        auto * registry = ggml_backend_dev_backend_reg(selected);
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_threads == nullptr) {
            return std::unexpected(fail(error_code::backend_unavailable,
                "loaded CPU backend does not expose thread configuration"));
        }
        set_threads(result_handle->value, static_cast<int>(bounded));
    }
    return result_handle;
}

} // namespace aam::detail

namespace aam {

std::vector<std::string> available_backends(const std::filesystem::path & backend_directory) {
    detail::load_backends(backend_directory);

    std::vector<std::string> names;
    names.reserve(ggml_backend_dev_count());
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        const char * description = ggml_backend_dev_description(device);
        names.emplace_back(description == nullptr ? ggml_backend_dev_name(device) : description);
    }
    return names;
}

} // namespace aam
