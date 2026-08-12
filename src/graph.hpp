#pragma once

#include "model_impl.hpp"

#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aam::detail {

class graph_arena final {
public:
    explicit graph_arena(std::size_t capacity);
    graph_arena(const graph_arena &) = delete;
    graph_arena & operator=(const graph_arena &) = delete;
    ~graph_arena();

    [[nodiscard]] bool valid() const noexcept { return context_ != nullptr && graph_ != nullptr; }
    [[nodiscard]] ggml_context * context() const noexcept { return context_; }
    [[nodiscard]] ggml_cgraph * graph() const noexcept { return graph_; }

private:
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class graph_ops final {
public:
    graph_ops(ggml_context * context,
              const std::unordered_map<std::string, ggml_tensor *> & tensors)
        : context_(context), tensors_(tensors) {}

    [[nodiscard]] ggml_tensor * weight(const std::string & name);
    [[nodiscard]] ggml_tensor * optional_weight(const std::string & name) const;
    [[nodiscard]] ggml_tensor * linear(ggml_tensor * input, const std::string & prefix);
    [[nodiscard]] ggml_tensor * layer_norm(
        ggml_tensor * input, const std::string & prefix, float epsilon);
    [[nodiscard]] ggml_tensor * attention(
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        std::int64_t width, std::int64_t heads,
        std::int64_t query_tokens, std::int64_t key_tokens,
        std::int64_t batch, ggml_tensor * mask = nullptr);
    [[nodiscard]] ggml_tensor * attention_values(
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        std::int64_t key_width, std::int64_t value_width, std::int64_t heads,
        std::int64_t query_tokens, std::int64_t key_tokens,
        std::int64_t batch, ggml_tensor * mask = nullptr);

    [[nodiscard]] const std::string & missing() const noexcept { return missing_; }

private:
    ggml_context * context_;
    const std::unordered_map<std::string, ggml_tensor *> & tensors_;
    std::string missing_;
};

struct upload {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    std::size_t bytes = 0;
};

struct download {
    ggml_tensor * tensor = nullptr;
    void * data = nullptr;
    std::size_t bytes = 0;
};

[[nodiscard]] result<void> execute_graph(
    ggml_backend_t backend,
    ggml_gallocr_t allocator,
    ggml_cgraph * graph,
    std::span<const upload> uploads,
    std::span<const download> downloads = {});

} // namespace aam::detail
