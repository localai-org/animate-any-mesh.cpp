#include "model_impl.hpp"
#include "pytorch_rng.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numbers>
#include <random>
namespace aam {

model::impl::~impl() {
    if (compute_allocator != nullptr) ggml_gallocr_free(compute_allocator);
    if (weights_buffer != nullptr) ggml_backend_buffer_free(weights_buffer);
    backend.reset();
    if (gguf != nullptr) gguf_free(gguf);
    if (weights_context != nullptr) ggml_free(weights_context);
}

namespace {

bool has_f32_fps_weights(
    const std::unordered_map<std::string, ggml_tensor *> & tensors) {
    constexpr std::string_view names[] = {
        "v.pe.m.w",
        "v.nl.0.fn.q.w", "v.nl.0.fn.k.w", "v.nl.0.fn.v.w",
        "v.nl.0.fn.o.w",
    };
    return std::ranges::all_of(names, [&](std::string_view name) {
        const auto found = tensors.find(std::string{name});
        return found != tensors.end() && found->second->type == GGML_TYPE_F32;
    });
}

class stage_profiler final {
public:
    stage_profiler() noexcept
        : enabled_(std::getenv("AAM_PROFILE") != nullptr),
          started_(clock::now()), previous_(started_) {}

    void checkpoint(const char * name) noexcept {
        if (!enabled_) return;
        const auto now = clock::now();
        const std::chrono::duration<double, std::milli> elapsed = now - previous_;
        std::fprintf(stderr, "aam_profile stage=%s ms=%.3f\n", name, elapsed.count());
        previous_ = now;
    }

    void finish() noexcept {
        if (!enabled_) return;
        const std::chrono::duration<double, std::milli> elapsed = clock::now() - started_;
        std::fprintf(stderr, "aam_profile stage=generate_total ms=%.3f\n", elapsed.count());
    }

private:
    using clock = std::chrono::steady_clock;
    bool enabled_;
    clock::time_point started_;
    clock::time_point previous_;
};

[[nodiscard]] const char * metadata_string(const gguf_context * context, const char * key) {
    const auto index = gguf_find_key(context, key);
    return index < 0 ? nullptr : gguf_get_val_str(context, index);
}

class normal_generator final {
public:
    explicit normal_generator(std::uint64_t seed) : engine_(seed) {}

    [[nodiscard]] float normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        constexpr double unit = 1.0 / 9007199254740992.0; // 2^-53
        const double first = (static_cast<double>(engine_() >> 11U) + 1.0) * unit;
        const double second = static_cast<double>(engine_() >> 11U) * unit;
        const double radius = std::sqrt(-2.0 * std::log(first));
        const double angle = 2.0 * std::numbers::pi * second;
        spare_ = static_cast<float>(radius * std::sin(angle));
        has_spare_ = true;
        return static_cast<float>(radius * std::cos(angle));
    }

private:
    std::mt19937_64 engine_;
    float spare_ = 0.0F;
    bool has_spare_ = false;
};

} // namespace

result<void> model::impl::stream_weights(const std::filesystem::path & path) {
    weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context, backend->value);
    if (weights_buffer == nullptr) {
        return std::unexpected(detail::fail(error_code::allocation,
            "failed to allocate GGUF tensors on " + backend->name));
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::unexpected(detail::fail(error_code::io, "cannot reopen GGUF weight file"));
    }
    const auto data_offset = gguf_get_data_offset(gguf);
    const auto tensor_count = gguf_get_n_tensors(gguf);
    std::vector<std::byte> staging;
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        const char * name = gguf_get_tensor_name(gguf, index);
        auto tensor_it = tensors.find(name == nullptr ? "" : name);
        if (tensor_it == tensors.end()) {
            return std::unexpected(detail::fail(error_code::incompatible_model,
                "GGUF tensor table is internally inconsistent"));
        }
        auto * tensor = tensor_it->second;
        const auto bytes = ggml_nbytes(tensor);
        const auto relative_offset = gguf_get_tensor_offset(gguf, index);
        if (relative_offset > std::numeric_limits<std::size_t>::max() - data_offset) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "GGUF tensor offset overflows address space"));
        }
        const auto absolute_offset = data_offset + relative_offset;
        if (absolute_offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()) ||
            bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "GGUF tensor is too large for the host stream API"));
        }
        staging.resize(bytes);
        stream.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
        if (!stream || !stream.read(reinterpret_cast<char *>(staging.data()),
                                    static_cast<std::streamsize>(bytes))) {
            return std::unexpected(detail::fail(error_code::io,
                "failed to read GGUF tensor '" + std::string{name} + "'"));
        }
        ggml_backend_tensor_set(tensor, staging.data(), 0, bytes);
    }
    return {};
}

model::model(std::unique_ptr<impl> value) noexcept : impl_(std::move(value)) {}
model::model(model &&) noexcept = default;
model & model::operator=(model &&) noexcept = default;
model::~model() = default;

result<model> model::load(const std::filesystem::path & gguf_path, const runtime_options & options) {
    auto value = std::make_unique<impl>();
    gguf_init_params parameters{};
    parameters.no_alloc = true;
    parameters.ctx = &value->weights_context;
    value->gguf = gguf_init_from_file(gguf_path.string().c_str(), parameters);
    if (value->gguf == nullptr || value->weights_context == nullptr) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "failed to parse GGUF model"));
    }
    const char * architecture = metadata_string(value->gguf, "general.architecture");
    if (architecture == nullptr || std::string_view{architecture} != "animate-any-mesh") {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "GGUF architecture is not 'animate-any-mesh'"));
    }
    for (auto * tensor = ggml_get_first_tensor(value->weights_context); tensor != nullptr;
         tensor = ggml_get_next_tensor(value->weights_context, tensor)) {
        value->tensors.emplace(tensor->name, tensor);
    }
    auto tokenizer_status = value->tokenizer.load(value->gguf);
    if (!tokenizer_status) return std::unexpected(tokenizer_status.error());
    auto backend = detail::make_backend(options);
    if (!backend) return std::unexpected(backend.error());
    value->backend = std::move(*backend);
    value->compute_allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(value->backend->value));
    if (value->compute_allocator == nullptr) {
        return std::unexpected(detail::fail(error_code::allocation,
            "failed to create reusable GGML graph allocator"));
    }
    auto weights_status = value->stream_weights(gguf_path);
    if (!weights_status) return std::unexpected(weights_status.error());
    const auto device_type = ggml_backend_dev_type(
        ggml_backend_get_device(value->backend->value));
    if (options.strict_fps_parity && device_type == GGML_BACKEND_DEVICE_TYPE_GPU &&
        has_f32_fps_weights(value->tensors)) {
        runtime_options cpu_options = options;
        cpu_options.device = device_kind::cpu;
        cpu_options.strict_fps_parity = false;
        auto cpu_reference = model::load(gguf_path, cpu_options);
        if (!cpu_reference) return std::unexpected(cpu_reference.error());
        value->fps_reference = std::move(cpu_reference->impl_);
    }
    return model{std::move(value)};
}

result<animation> model::generate(
    const mesh & source, std::string_view prompt, const generation_options & options) const {
    if (impl_ == nullptr) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "model is empty"));
    }
    auto prepared = prepare_mesh(source, options.max_vertices);
    if (!prepared) return std::unexpected(prepared.error());
    return generate(*prepared, prompt, options);
}

result<animation> model::generate(
    const prepared_mesh & source, std::string_view prompt,
    const generation_options & options) const {
    stage_profiler profile;
    if (impl_ == nullptr) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "model is empty"));
    }
    if (source.value.vertices.empty() || source.original_to_merged.empty() ||
        std::ranges::any_of(source.original_to_merged, [&](std::uint32_t value) {
            return value >= source.value.vertices.size();
        })) {
        return std::unexpected(detail::fail(
            error_code::invalid_argument, "invalid prepared mesh mapping"));
    }
    auto tokens = impl_->tokenizer.encode(prompt);
    if (!tokens) return std::unexpected(tokens.error());
    if (options.steps < 2U || options.steps > 1024U ||
        !std::isfinite(options.guidance_scale)) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "sampling steps must be in [2, 1024] and guidance must be finite"));
    }
    constexpr std::size_t maximum_trajectories = 16384U;
    std::size_t trajectory_count = options.trajectories;
    if (trajectory_count == 0U) {
        trajectory_count = std::max<std::size_t>(
            512U, source.value.vertices.size() / 8U);
    }
    if (trajectory_count > maximum_trajectories) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "trajectory count exceeds the safety limit of 16384"));
    }
    profile.checkpoint("tokenize_validate");

    auto embeddings = impl_->vae_embed(
        source.value.vertices, source.adjacency_offsets,
        source.adjacency_indices);
    if (!embeddings) return std::unexpected(embeddings.error());
    profile.checkpoint("vae_embed");
    const detail::vae_embeddings * sampling_embeddings = &*embeddings;
    std::optional<detail::vae_embeddings> reference_embeddings;
    if (impl_->fps_reference != nullptr) {
        auto reference = impl_->fps_reference->vae_embed(
            source.value.vertices, source.adjacency_offsets,
            source.adjacency_indices);
        if (!reference) return std::unexpected(reference.error());
        reference_embeddings = std::move(*reference);
        sampling_embeddings = &*reference_embeddings;
        profile.checkpoint("vae_embed_cpu_fps_reference");
    }
    const auto sampled_count = std::min(trajectory_count, embeddings->points);
    auto sampled = farthest_point_sample(
        sampling_embeddings->point, sampling_embeddings->points, 256U, sampled_count);
    if (!sampled) return std::unexpected(sampled.error());
    profile.checkpoint("farthest_point_sample");

    normal_generator posterior_random{options.seed};
    detail::pytorch_cuda_generator sampling_random{options.seed};
    std::vector<std::uint32_t> indices(trajectory_count, 0U);
    std::copy(sampled->begin(), sampled->end(), indices.begin());
    // The official replace_negative_indices treats index zero like PyTorch3D's
    // -1 padding. Preserve that published inference behavior for parity.
    for (std::size_t position = 0U; position < indices.size(); ++position) {
        auto & index = indices[position];
        if (index == 0U) {
            index = static_cast<std::uint32_t>(
                sampling_random.replacement_index(position, embeddings->points));
        }
    }
    std::vector<float> epsilon(trajectory_count * 32U);
    std::ranges::generate(epsilon, [&] { return posterior_random.normal(); });
    auto encoded = impl_->vae_encode(*embeddings, indices, epsilon);
    if (!encoded) return std::unexpected(encoded.error());
    profile.checkpoint("vae_encode");

    constexpr float f0_mean = 0.205227330327034F;
    constexpr float f0_std = 9.064417839050293F;
    constexpr float ft_mean = 0.95684415102005F;
    constexpr float ft_std = 6.397393703460693F;
    std::vector<float> normalized = encoded->latent;
    std::vector<float> f0(trajectory_count * 32U);
    for (std::size_t token = 0; token < trajectory_count; ++token) {
        for (std::size_t channel = 0; channel < 64U; ++channel) {
            auto & value = normalized[token * 64U + channel];
            if (channel < 32U) {
                value = (value - f0_mean) / f0_std;
                f0[token * 32U + channel] = value;
            } else {
                value = (value - ft_mean) / ft_std;
            }
        }
    }
    profile.checkpoint("normalize");

    auto empty_tokens = impl_->tokenizer.encode("");
    if (!empty_tokens) return std::unexpected(empty_tokens.error());
    std::vector<std::int32_t> clip_tokens;
    clip_tokens.reserve(154U);
    clip_tokens.insert(clip_tokens.end(), tokens->begin(), tokens->end());
    clip_tokens.insert(clip_tokens.end(), empty_tokens->begin(), empty_tokens->end());
    auto clip = impl_->clip_encode(clip_tokens, 2U);
    if (!clip) return std::unexpected(clip.error());
    profile.checkpoint("clip_encode");

    std::vector<float> initial_noise(trajectory_count * 64U);
    for (std::size_t index = 0U; index < initial_noise.size(); ++index) {
        initial_noise[index] = sampling_random.normal(index, initial_noise.size());
    }
    profile.checkpoint("initial_noise");
    auto sampled_latent = impl_->rf_sample(
        f0, trajectory_count, *clip, initial_noise,
        options.steps, options.guidance_scale);
    if (!sampled_latent) return std::unexpected(sampled_latent.error());
    profile.checkpoint("rf_sample");
    for (std::size_t token = 0; token < trajectory_count; ++token) {
        for (std::size_t channel = 0; channel < 64U; ++channel) {
            auto & value = (*sampled_latent)[token * 64U + channel];
            value = channel < 32U ? std::fma(value, f0_std, f0_mean)
                                  : std::fma(value, ft_std, ft_mean);
        }
    }
    profile.checkpoint("denormalize");
    auto offsets = impl_->vae_decode_offsets(
        *embeddings, *sampled_latent, trajectory_count);
    if (!offsets) return std::unexpected(offsets.error());
    profile.checkpoint("vae_decode");

    animation output;
    output.frames = default_frames;
    output.vertices_per_frame = source.original_to_merged.size();
    output.positions.resize(output.frames * output.vertices_per_frame);
    for (std::size_t frame = 0; frame < output.frames; ++frame) {
        for (std::size_t original = 0; original < source.original_to_merged.size(); ++original) {
            const auto merged = static_cast<std::size_t>(source.original_to_merged[original]);
            const auto & base = source.value.vertices[merged];
            const std::size_t offset = merged * 48U + frame * 3U;
            auto & destination = output.positions[frame * output.vertices_per_frame + original];
            destination = {
                std::fma(base.x + (*offsets)[offset], source.transform.scale,
                         source.transform.center.x),
                std::fma(base.y + (*offsets)[offset + 1U], source.transform.scale,
                         source.transform.center.y),
                std::fma(base.z + (*offsets)[offset + 2U], source.transform.scale,
                         source.transform.center.z),
            };
        }
    }
    profile.checkpoint("assemble_animation");
    profile.finish();
    return output;
}

result<std::vector<std::int32_t>> model::tokenize(std::string_view prompt) const {
    if (impl_ == nullptr) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "model is empty"));
    }
    return impl_->tokenizer.encode(prompt);
}

result<text_embedding> model::encode_text(std::string_view prompt) const {
    auto token_ids = tokenize(prompt);
    if (!token_ids) return std::unexpected(token_ids.error());
    auto values = impl_->clip_encode(*token_ids, 1U);
    if (!values) return std::unexpected(values.error());
    return text_embedding{77U, 768U, std::move(*values)};
}

result<std::vector<text_embedding>> model::encode_text_blocks(std::string_view prompt) const {
    auto token_ids = tokenize(prompt);
    if (!token_ids) return std::unexpected(token_ids.error());
    std::vector<std::vector<float>> block_values;
    auto final_values = impl_->clip_encode(*token_ids, 1U, &block_values);
    if (!final_values) return std::unexpected(final_values.error());
    std::vector<text_embedding> result_values;
    result_values.reserve(block_values.size() + 1U);
    for (auto & values : block_values) {
        result_values.push_back({77U, 768U, std::move(values)});
    }
    result_values.push_back({77U, 768U, std::move(*final_values)});
    return result_values;
}

std::string_view model::backend_name() const noexcept {
    return impl_ == nullptr || impl_->backend == nullptr ? std::string_view{} : impl_->backend->name;
}

} // namespace aam
