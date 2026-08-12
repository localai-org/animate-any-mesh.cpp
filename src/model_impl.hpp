#pragma once

#include "internal.hpp"
#include "tokenizer.hpp"

#include <mutex>
#include <optional>
#include <unordered_map>

namespace aam {

namespace detail {

struct vae_embeddings {
    std::size_t points = 0;
    std::vector<float> point;
    std::vector<float> trajectory;
};

struct vae_embedding_trace {
    std::vector<float> point_initial;
};

struct vae_encoded {
    std::size_t trajectories = 0;
    std::vector<float> latent;
};

struct rf_trace {
    std::vector<float> time_projected;
    std::vector<float> text_projected;
    std::vector<float> x_initial;
    std::vector<float> text_initial;
    // Four entries per block: attention x/text, then feed-forward x/text.
    std::vector<std::vector<float>> blocks;
    std::vector<float> final_norm;
};

} // namespace detail

struct model::impl {
    gguf_context * gguf = nullptr;
    ggml_context * weights_context = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_gallocr_t compute_allocator = nullptr;
    std::unique_ptr<detail::backend_handle> backend;
    detail::clip_tokenizer tokenizer;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    mutable std::mutex compute_mutex;
    std::unique_ptr<impl> fps_reference;

    impl() = default;
    impl(const impl &) = delete;
    impl & operator=(const impl &) = delete;
    ~impl();

    [[nodiscard]] result<void> stream_weights(const std::filesystem::path & path);
    [[nodiscard]] result<std::vector<float>> clip_encode(
        std::span<const std::int32_t> tokens, std::size_t batch,
        std::vector<std::vector<float>> * block_outputs = nullptr) const;
    [[nodiscard]] result<detail::vae_embeddings> vae_embed(
        std::span<const vec3> points,
        std::span<const std::uint32_t> adjacency_offsets,
        std::span<const std::uint32_t> adjacency_indices,
        detail::vae_embedding_trace * trace = nullptr) const;
    [[nodiscard]] result<detail::vae_encoded> vae_encode(
        const detail::vae_embeddings & embeddings,
        std::span<const std::uint32_t> indices,
        std::span<const float> epsilon,
        std::vector<std::vector<float>> * block_outputs = nullptr) const;
    [[nodiscard]] result<std::vector<float>> vae_decode_offsets(
        const detail::vae_embeddings & embeddings,
        std::span<const float> latent,
        std::size_t trajectories,
        std::vector<std::vector<float>> * block_outputs = nullptr) const;
    [[nodiscard]] result<std::vector<float>> rf_forward(
        std::span<const float> input, std::size_t token_count,
        std::span<const float> clip, std::span<const float> times,
        std::span<const float> condition_mask,
        detail::rf_trace * trace = nullptr,
        std::optional<float> guidance_scale = std::nullopt) const;
    [[nodiscard]] result<std::vector<float>> rf_sample(
        std::span<const float> f0, std::size_t trajectories,
        std::span<const float> clip, std::span<const float> initial_noise,
        std::size_t steps, float guidance_scale) const;
};

} // namespace aam
