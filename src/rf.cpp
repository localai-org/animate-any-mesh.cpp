#include "graph.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace aam {
namespace {

struct adaln_parameters {
    ggml_tensor * normalized = nullptr;
    ggml_tensor * attention_gate = nullptr;
    ggml_tensor * ffn_shift = nullptr;
    ggml_tensor * ffn_scale = nullptr;
    ggml_tensor * ffn_gate = nullptr;
};

ggml_tensor * modulate(
    ggml_context * context, ggml_tensor * value,
    ggml_tensor * scale, ggml_tensor * shift) {
    if (value == nullptr || scale == nullptr || shift == nullptr) return nullptr;
    return ggml_add(context,
        ggml_add(context, value, ggml_mul(context, value, scale)), shift);
}

adaln_parameters adaln(
    ggml_context * context, detail::graph_ops & ops,
    ggml_tensor * input, ggml_tensor * silu_time,
    const std::string & prefix, std::int64_t width,
    std::int64_t batch) {
    auto * parameters = ops.linear(silu_time, prefix + ".linear");
    if (input == nullptr || parameters == nullptr) return {};
    auto chunk = [&](std::int64_t index) {
        return ggml_view_3d(
            context, parameters, width, 1, batch,
            static_cast<std::size_t>(width) * parameters->nb[0], parameters->nb[1],
            static_cast<std::size_t>(index * width) * parameters->nb[0]);
    };
    auto * normalized = ggml_norm(context, input, 1.0e-6F);
    normalized = modulate(context, normalized, chunk(1), chunk(0));
    return {normalized, chunk(2), chunk(3), chunk(4), chunk(5)};
}

ggml_tensor * joint_attention(
    ggml_context * context, detail::graph_ops & ops,
    ggml_tensor * input, const std::string & prefix,
    std::int64_t width, std::int64_t heads,
    std::int64_t tokens, std::int64_t batch) {
    auto * qkv = ops.linear(input, prefix + ".c_qkv");
    if (qkv == nullptr) return nullptr;
    const auto head_width = width / heads;
    auto * by_head = ggml_reshape_4d(context, qkv, 3 * head_width, heads, tokens, batch);
    auto strided_stream = [&](std::int64_t index) {
        auto * view = ggml_view_4d(
            context, by_head, head_width, heads, tokens, batch,
            by_head->nb[1], by_head->nb[2], by_head->nb[3],
            static_cast<std::size_t>(index * head_width) * by_head->nb[0]);
        return ggml_permute(context, view, 0, 2, 1, 3);
    };
    if (std::getenv("AAM_EXACT_ATTENTION") == nullptr) {
        auto * attended = ggml_flash_attn_ext(
            context, strided_stream(0), strided_stream(1), strided_stream(2),
            nullptr, 1.0F / std::sqrt(static_cast<float>(head_width)), 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        return ops.linear(
            ggml_reshape_3d(context, attended, width, tokens, batch),
            prefix + ".c_proj");
    }
    auto contiguous_stream = [&](std::int64_t index) {
        auto * view = ggml_view_4d(
            context, by_head, head_width, heads, tokens, batch,
            by_head->nb[1], by_head->nb[2], by_head->nb[3],
            static_cast<std::size_t>(index * head_width) * by_head->nb[0]);
        return ggml_reshape_3d(context, ggml_cont(context, view), width, tokens, batch);
    };
    auto * attended = ops.attention(
        contiguous_stream(0), contiguous_stream(1), contiguous_stream(2),
        width, heads, tokens, tokens, batch);
    return ops.linear(attended, prefix + ".c_proj");
}

ggml_tensor * exact_gelu_mlp(
    ggml_context * context, detail::graph_ops & ops,
    ggml_tensor * input, const std::string & prefix) {
    auto * expanded = ops.linear(input, prefix + ".c_fc");
    if (expanded != nullptr) expanded = ggml_gelu_erf(context, expanded);
    return ops.linear(expanded, prefix + ".c_proj");
}

} // namespace

result<std::vector<float>> model::impl::rf_forward(
    std::span<const float> input, std::size_t token_count,
    std::span<const float> clip, std::span<const float> times,
    std::span<const float> condition_mask, detail::rf_trace * trace,
    std::optional<float> guidance_scale) const {
    constexpr std::int64_t input_width = 64;
    constexpr std::int64_t width = 512;
    constexpr std::int64_t clip_width = 768;
    constexpr std::int64_t text_tokens = 77;
    constexpr std::int64_t heads = 8;
    constexpr std::int64_t layers = 12;
    const std::size_t batch_size = times.size();
    if (batch_size == 0U || token_count == 0U ||
        batch_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        token_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        input.size() != batch_size * token_count * static_cast<std::size_t>(input_width) ||
        clip.size() != batch_size * static_cast<std::size_t>(text_tokens * clip_width) ||
        condition_mask.size() != batch_size ||
        (guidance_scale.has_value() &&
         (batch_size != 2U || !std::isfinite(*guidance_scale)))) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "invalid rectified-flow input dimensions"));
    }
    const auto batch = static_cast<std::int64_t>(batch_size);
    const auto x_tokens = static_cast<std::int64_t>(token_count);
    const auto all_tokens = x_tokens + text_tokens;

    std::vector<float> time_embedding(batch_size * static_cast<std::size_t>(width));
    const float log_period = std::log(10000.0F);
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::int64_t index = 0; index < width / 2; ++index) {
            const float frequency = std::exp(
                -log_period * static_cast<float>(index) / static_cast<float>(width / 2));
            const float argument = times[b] * frequency;
            time_embedding[b * static_cast<std::size_t>(width) + static_cast<std::size_t>(index)] =
                std::cos(argument);
            time_embedding[b * static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(index + width / 2)] = std::sin(argument);
        }
    }

    std::scoped_lock lock{compute_mutex};
    detail::graph_arena arena{65536U};
    if (!arena.valid()) return std::unexpected(detail::fail(
        error_code::allocation, "failed to create rectified-flow graph"));
    auto * context = arena.context();
    auto * graph = arena.graph();
    detail::graph_ops ops{context, tensors};
    auto * x_input = ggml_new_tensor_3d(context, GGML_TYPE_F32, input_width, x_tokens, batch);
    auto * clip_input = ggml_new_tensor_3d(context, GGML_TYPE_F32, clip_width, text_tokens, batch);
    auto * time_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, batch);
    auto * mask_input = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 1, batch);
    ggml_set_input(x_input); ggml_set_input(clip_input);
    ggml_set_input(time_input); ggml_set_input(mask_input);

    std::vector<ggml_tensor *> captured;
    auto capture = [&](ggml_tensor * value) {
        if (trace == nullptr || value == nullptr) return value;
        auto * contiguous = ggml_cont(context, value);
        ggml_set_output(contiguous);
        ggml_build_forward_expand(graph, contiguous);
        captured.push_back(contiguous);
        return contiguous;
    };

    auto * projected_time = exact_gelu_mlp(context, ops, time_input, "rf.time_embed");
    capture(projected_time);
    auto * silu_time = projected_time == nullptr ? nullptr : ggml_silu(context, projected_time);
    auto * text = ops.linear(clip_input, "rf.clip_token_mlp");
    text = text == nullptr ? nullptr : ggml_mul(context, text, mask_input);
    capture(text);
    auto * x = ops.linear(x_input, "rf.input_proj");
    x = ops.layer_norm(x, "rf.ln_pre", 1.0e-5F);
    text = ops.layer_norm(text, "rf.ln_pre_text", 1.0e-5F);
    capture(x); capture(text);

    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "rf.backbone.resblocks." + std::to_string(layer);
        const auto x_parameters = adaln(
            context, ops, x, silu_time, prefix + ".adaln_x", width, batch);
        const auto text_parameters = adaln(
            context, ops, text, silu_time, prefix + ".adaln_text", width, batch);
        auto * combined = ggml_concat(
            context, x_parameters.normalized, text_parameters.normalized, 1);
        auto * attention = joint_attention(
            context, ops, combined, prefix + ".attn", width, heads, all_tokens, batch);
        auto * attention_x = attention == nullptr ? nullptr : ggml_view_3d(
            context, attention, width, x_tokens, batch,
            attention->nb[1], attention->nb[2], 0);
        auto * attention_text = attention == nullptr ? nullptr : ggml_view_3d(
            context, attention, width, text_tokens, batch,
            attention->nb[1], attention->nb[2],
            static_cast<std::size_t>(x_tokens) * attention->nb[1]);
        x = x == nullptr || attention_x == nullptr ? nullptr
            : ggml_add(context, x, ggml_mul(context, attention_x, x_parameters.attention_gate));
        text = text == nullptr || attention_text == nullptr ? nullptr
            : ggml_add(context, text, ggml_mul(context, attention_text, text_parameters.attention_gate));
        capture(x); capture(text);

        auto * normalized_x = ops.layer_norm(x, prefix + ".ln_x", 1.0e-5F);
        auto * normalized_text = ops.layer_norm(text, prefix + ".ln_text", 1.0e-5F);
        normalized_x = modulate(
            context, normalized_x, x_parameters.ffn_scale, x_parameters.ffn_shift);
        normalized_text = modulate(
            context, normalized_text, text_parameters.ffn_scale, text_parameters.ffn_shift);
        auto * ffn = exact_gelu_mlp(context, ops,
            ggml_concat(context, normalized_x, normalized_text, 1), prefix + ".mlp");
        auto * ffn_x = ffn == nullptr ? nullptr : ggml_view_3d(
            context, ffn, width, x_tokens, batch, ffn->nb[1], ffn->nb[2], 0);
        auto * ffn_text = ffn == nullptr ? nullptr : ggml_view_3d(
            context, ffn, width, text_tokens, batch, ffn->nb[1], ffn->nb[2],
            static_cast<std::size_t>(x_tokens) * ffn->nb[1]);
        x = x == nullptr || ffn_x == nullptr ? nullptr
            : ggml_add(context, x, ggml_mul(context, ffn_x, x_parameters.ffn_gate));
        text = text == nullptr || ffn_text == nullptr ? nullptr
            : ggml_add(context, text, ggml_mul(context, ffn_text, text_parameters.ffn_gate));
        capture(x); capture(text);
    }
    auto * final_norm = ops.layer_norm(x, "rf.ln_post", 1.0e-5F);
    capture(final_norm);
    auto * output = ops.linear(final_norm, "rf.output_proj");
    if (!ops.missing().empty() || output == nullptr) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "missing rectified-flow tensor '" + ops.missing() + "'"));
    }
    if (guidance_scale.has_value()) {
        auto * conditioned = ggml_view_2d(
            context, output, input_width, x_tokens, output->nb[1], 0);
        auto * unconditioned = ggml_view_2d(
            context, output, input_width, x_tokens, output->nb[1], output->nb[2]);
        output = ggml_add(context, unconditioned,
            ggml_scale(context, ggml_sub(context, conditioned, unconditioned),
                       *guidance_scale));
    }
    output = ggml_cont(context, output);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    const std::array uploads{
        detail::upload{x_input, input.data(), input.size_bytes()},
        detail::upload{clip_input, clip.data(), clip.size_bytes()},
        detail::upload{time_input, time_embedding.data(), time_embedding.size() * sizeof(float)},
        detail::upload{mask_input, condition_mask.data(), condition_mask.size_bytes()},
    };
    const auto result_count = guidance_scale.has_value()
        ? token_count * static_cast<std::size_t>(input_width) : input.size();
    std::vector<float> result_values(result_count);
    std::vector<std::vector<float>> captured_values(captured.size());
    std::vector<detail::download> downloads;
    downloads.reserve(captured.size() + 1U);
    for (std::size_t index = 0; index < captured.size(); ++index) {
        captured_values[index].resize(static_cast<std::size_t>(ggml_nelements(captured[index])));
        downloads.push_back({captured[index], captured_values[index].data(),
                             captured_values[index].size() * sizeof(float)});
    }
    downloads.push_back({output, result_values.data(), result_values.size() * sizeof(float)});
    auto status = detail::execute_graph(
        backend->value, compute_allocator, graph, uploads, downloads);
    if (!status) return std::unexpected(status.error());
    if (trace != nullptr) {
        if (captured_values.size() != 53U) {
            return std::unexpected(detail::fail(error_code::compute,
                "rectified-flow trace has an internal stage-count mismatch"));
        }
        trace->time_projected = std::move(captured_values[0]);
        trace->text_projected = std::move(captured_values[1]);
        trace->x_initial = std::move(captured_values[2]);
        trace->text_initial = std::move(captured_values[3]);
        trace->blocks.clear();
        trace->blocks.reserve(48U);
        for (std::size_t index = 4; index < 52U; ++index) {
            trace->blocks.push_back(std::move(captured_values[index]));
        }
        trace->final_norm = std::move(captured_values[52]);
    }
    return result_values;
}

result<std::vector<float>> model::impl::rf_sample(
    std::span<const float> f0, std::size_t trajectories,
    std::span<const float> clip, std::span<const float> initial_noise,
    std::size_t steps, float guidance_scale) const {
    constexpr std::size_t latent_width = 64U;
    constexpr std::size_t static_width = 32U;
    constexpr std::size_t clip_batch_values = 77U * 768U;
    if (trajectories == 0U ||
        trajectories > std::numeric_limits<std::size_t>::max() / latent_width ||
        steps < 2U || steps > 1024U ||
        !std::isfinite(guidance_scale) ||
        f0.size() != trajectories * static_width ||
        initial_noise.size() != trajectories * latent_width ||
        clip.size() != 2U * clip_batch_values) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "invalid rectified-flow sampling parameters"));
    }
    std::vector<float> state(initial_noise.begin(), initial_noise.end());
    std::vector<float> midpoint(state.size());
    std::vector<float> model_input(2U * state.size());
    const std::array<float, 2> condition_mask{1.0F, 0.0F};
    const float delta = 1.0F / static_cast<float>(steps - 1U);
    const bool profiling = std::getenv("AAM_PROFILE") != nullptr;
    using profile_clock = std::chrono::steady_clock;
    const auto profile_started = profile_clock::now();
    std::chrono::duration<double, std::milli> input_time{};
    std::chrono::duration<double, std::milli> forward_time{};
    std::chrono::duration<double, std::milli> validation_time{};
    std::chrono::duration<double, std::milli> solver_time{};
    std::size_t evaluations = 0U;

    auto velocity = [&](std::span<const float> value, float time)
        -> result<std::vector<float>> {
        const auto input_started = profile_clock::now();
        for (std::size_t batch = 0; batch < 2U; ++batch) {
            const std::size_t batch_offset = batch * state.size();
            std::copy(value.begin(), value.end(),
                      model_input.begin() + static_cast<std::ptrdiff_t>(batch_offset));
            for (std::size_t token = 0; token < trajectories; ++token) {
                const std::size_t token_offset = token * latent_width;
                std::copy_n(f0.begin() + static_cast<std::ptrdiff_t>(token * static_width),
                            static_width,
                            model_input.begin() + static_cast<std::ptrdiff_t>(
                batch_offset + token_offset));
            }
        }
        if (profiling) input_time += profile_clock::now() - input_started;
        const std::array times{time, time};
        const auto forward_started = profile_clock::now();
        auto prediction = rf_forward(
            model_input, trajectories, clip, times, condition_mask, nullptr,
            guidance_scale);
        if (profiling) forward_time += profile_clock::now() - forward_started;
        if (!prediction) return std::unexpected(prediction.error());
        const auto validation_started = profile_clock::now();
        for (const float predicted : *prediction) {
            if (!std::isfinite(predicted)) {
                return std::unexpected(detail::fail(error_code::compute,
                    "rectified-flow model produced NaN or infinity"));
            }
        }
        if (profiling) validation_time += profile_clock::now() - validation_started;
        ++evaluations;
        return prediction;
    };

    // torchdiffeq's fixed-grid midpoint solver performs two model evaluations
    // per interval and returns all requested grid points. Only the final point
    // is retained here, avoiding 64 copies of the full latent trajectory.
    for (std::size_t step = 0; step + 1U < steps; ++step) {
        const float time = static_cast<float>(step) * delta;
        auto start_velocity = velocity(state, time);
        if (!start_velocity) return std::unexpected(start_velocity.error());
        const auto midpoint_started = profile_clock::now();
        for (std::size_t index = 0; index < state.size(); ++index) {
            midpoint[index] = std::fma(0.5F * delta, (*start_velocity)[index], state[index]);
        }
        if (profiling) solver_time += profile_clock::now() - midpoint_started;
        auto midpoint_velocity = velocity(midpoint, time + 0.5F * delta);
        if (!midpoint_velocity) return std::unexpected(midpoint_velocity.error());
        const auto state_started = profile_clock::now();
        for (std::size_t index = 0; index < state.size(); ++index) {
            state[index] = std::fma(delta, (*midpoint_velocity)[index], state[index]);
        }
        if (profiling) solver_time += profile_clock::now() - state_started;
    }
    for (std::size_t token = 0; token < trajectories; ++token) {
        std::copy_n(f0.begin() + static_cast<std::ptrdiff_t>(token * static_width),
                    static_width,
                    state.begin() + static_cast<std::ptrdiff_t>(token * latent_width));
    }
    if (profiling) {
        const std::chrono::duration<double, std::milli> total =
            profile_clock::now() - profile_started;
        std::fprintf(stderr,
            "aam_profile rf evaluations=%zu input_ms=%.3f forward_ms=%.3f "
            "validate_ms=%.3f solver_ms=%.3f total_ms=%.3f\n",
            evaluations, input_time.count(), forward_time.count(),
            validation_time.count(), solver_time.count(), total.count());
    }
    return state;
}

} // namespace aam
