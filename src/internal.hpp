#pragma once

#include "aam/aam.hpp"

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <gguf.h>
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <limits>
#include <string>
#include <string_view>

namespace aam::detail {

[[nodiscard]] inline error fail(error_code code, std::string message) {
    return error{code, std::move(message)};
}

template<class T>
[[nodiscard]] constexpr bool checked_mul(T a, T b, T & out) noexcept {
    static_assert(std::numeric_limits<T>::is_integer);
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        return false;
    }
    out = static_cast<T>(a * b);
    return true;
}

struct backend_handle {
    ggml_backend_t value = nullptr;
    std::string name;

    backend_handle() = default;
    backend_handle(const backend_handle &) = delete;
    backend_handle & operator=(const backend_handle &) = delete;
    ~backend_handle();
};

[[nodiscard]] result<std::unique_ptr<backend_handle>> make_backend(
    const runtime_options & options);

} // namespace aam::detail
