#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aam {
namespace {

ggml_tensor * geglu(ggml_context * context, ggml_tensor * input,
                    std::int64_t hidden, std::int64_t tokens, std::int64_t batch) {
    if (input == nullptr) return nullptr;
    auto * values = ggml_cont(context, ggml_view_3d(
        context, input, hidden, tokens, batch, input->nb[1], input->nb[2], 0));
    auto * gates = ggml_cont(context, ggml_view_3d(
        context, input, hidden, tokens, batch, input->nb[1], input->nb[2],
        static_cast<std::size_t>(hidden) * input->nb[0]));
    return ggml_mul(context, values, ggml_gelu_erf(context, gates));
}

std::pair<ggml_tensor *, ggml_tensor *> synchronized_attention(
    ggml_context * context, detail::graph_ops & ops,
    ggml_tensor * q_stream, ggml_tensor * k_stream,
    ggml_tensor * v1_stream, ggml_tensor * v2_stream,
    const std::string & prefix, std::int64_t width, std::int64_t inner,
    std::int64_t heads, std::int64_t query_tokens, std::int64_t key_tokens,
    std::int64_t batch) {
    auto * qn = ops.layer_norm(q_stream, prefix + ".norm_q", 1.0e-5F);
    auto * kn = ops.layer_norm(k_stream, prefix + ".norm_k", 1.0e-5F);
    auto * v1n = ops.layer_norm(v1_stream, prefix + ".norm_v1", 1.0e-5F);
    auto * v2n = ops.layer_norm(v2_stream, prefix + ".norm_v2", 1.0e-5F);
    auto * q = ops.linear(qn, prefix + ".attn.to_q");
    auto * k = ops.linear(kn, prefix + ".attn.to_k");
    auto * v1 = ops.linear(v1n, prefix + ".attn.to_v1");
    auto * v2 = ops.linear(v2n, prefix + ".attn.to_v2");
    // PyTorch concatenates the two value streams inside each attention head:
    // [head0.v1, head0.v2, head1.v1, head1.v2, ...]. A direct concatenation
    // along the flat feature axis instead produces [all v1, all v2], which is
    // only equivalent for a single head.
    const auto head_width = inner / heads;
    auto * v1_heads = v1 == nullptr ? nullptr
        : ggml_reshape_4d(context, v1, head_width, heads, key_tokens, batch);
    auto * v2_heads = v2 == nullptr ? nullptr
        : ggml_reshape_4d(context, v2, head_width, heads, key_tokens, batch);
    auto * values_by_head = v1_heads == nullptr || v2_heads == nullptr ? nullptr
        : ggml_concat(context, v1_heads, v2_heads, 0);
    auto * values = values_by_head == nullptr ? nullptr
        : ggml_reshape_3d(context, values_by_head, 2 * inner, key_tokens, batch);
    auto * combined = ops.attention_values(q, k, values, inner, 2 * inner, heads,
                                          query_tokens, key_tokens, batch);
    if (combined == nullptr) return {nullptr, nullptr};
    auto * combined_heads = ggml_reshape_4d(
        context, combined, 2 * head_width, heads, query_tokens, batch);
    auto stream_view = [&](std::size_t offset) {
        auto * view = ggml_view_4d(
            context, combined_heads, head_width, heads, query_tokens, batch,
            combined_heads->nb[1], combined_heads->nb[2], combined_heads->nb[3], offset);
        return ggml_reshape_3d(
            context, ggml_cont(context, view), inner, query_tokens, batch);
    };
    auto * first = stream_view(0U);
    auto * second = stream_view(static_cast<std::size_t>(head_width) * combined_heads->nb[0]);
    (void)width;
    return {ops.linear(first, prefix + ".attn.to_out1"),
            ops.linear(second, prefix + ".attn.to_out2")};
}

std::pair<ggml_tensor *, ggml_tensor *> synchronized_feed_forward(
    ggml_context * context, detail::graph_ops & ops,
    ggml_tensor * first, ggml_tensor * second,
    const std::string & prefix, std::int64_t width,
    std::int64_t tokens, std::int64_t batch) {
    auto * first_norm = ops.layer_norm(first, prefix + ".norm1", 1.0e-5F);
    auto * second_norm = ops.layer_norm(second, prefix + ".norm2", 1.0e-5F);
    auto stream = [&](ggml_tensor * value, const char * which) {
        auto * expanded = ops.linear(value, prefix + ".ffn." + which + ".0");
        auto * gated = geglu(context, expanded, width * 4, tokens, batch);
        return ops.linear(gated, prefix + ".ffn." + which + ".2");
    };
    return {stream(first_norm, "ffn1"), stream(second_norm, "ffn2")};
}

std::vector<ggml_fp16_t> adjacency_mask(
    std::size_t points, std::span<const std::uint32_t> offsets,
    std::span<const std::uint32_t> indices) {
    const auto negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    std::vector<ggml_fp16_t> result(points * points, negative_infinity);
    for (std::size_t query = 0; query < points; ++query) {
        for (std::size_t position = offsets[query]; position < offsets[query + 1U]; ++position) {
            const auto key = static_cast<std::size_t>(indices[position]);
            result[key + points * query] = ggml_fp32_to_fp16(0.0F);
        }
    }
    return result;
}

} // namespace

result<detail::vae_embeddings> model::impl::vae_embed(
    std::span<const vec3> points,
    std::span<const std::uint32_t> adjacency_offsets,
    std::span<const std::uint32_t> adjacency_indices,
    detail::vae_embedding_trace * trace) const {
    constexpr std::int64_t width = 256;
    constexpr std::int64_t neighbor_inner = 512;
    if (points.empty() || points.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        adjacency_offsets.size() != points.size() + 1U || adjacency_offsets.front() != 0U ||
        adjacency_offsets.back() != adjacency_indices.size()) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "invalid mesh adjacency CSR"));
    }
    for (std::size_t row = 0; row < points.size(); ++row) {
        if (adjacency_offsets[row] > adjacency_offsets[row + 1U] ||
            std::ranges::any_of(adjacency_indices.subspan(
                adjacency_offsets[row], adjacency_offsets[row + 1U] - adjacency_offsets[row]),
                [&](std::uint32_t value) { return value >= points.size(); })) {
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh adjacency CSR is out of bounds"));
        }
    }

    const auto count = static_cast<std::int64_t>(points.size());
    std::vector<float> trajectory_features(points.size() * 1008U, 0.0F);
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::fill_n(trajectory_features.begin() + static_cast<std::ptrdiff_t>(i * 1008U + 528U),
                    480U, 1.0F);
    }
    auto mask_values = adjacency_mask(points.size(), adjacency_offsets, adjacency_indices);

    std::scoped_lock lock{compute_mutex};
    detail::graph_arena arena{8192U};
    if (!arena.valid()) return std::unexpected(detail::fail(error_code::allocation, "failed to create VAE embed graph"));
    auto * context = arena.context();
    auto * graph = arena.graph();
    detail::graph_ops ops{context, tensors};
    auto * basis = ops.weight("dvae.point_embed.basis");
    constexpr std::size_t coordinate_width = 3U;
    constexpr std::size_t projection_width = 24U;
    constexpr std::size_t point_input_width = 2U * projection_width + coordinate_width;
    if (basis == nullptr || basis->type != GGML_TYPE_F32 ||
        basis->ne[0] != static_cast<std::int64_t>(projection_width) ||
        basis->ne[1] != static_cast<std::int64_t>(coordinate_width) ||
        ggml_nelements(basis) != coordinate_width * projection_width) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "VAE point embedding basis has an incompatible shape or type"));
    }
    std::array<float, coordinate_width * projection_width> basis_values{};
    ggml_backend_tensor_get(basis, basis_values.data(), 0, sizeof(basis_values));
    std::vector<float> point_embedding_features(points.size() * point_input_width);
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        const auto & point = points[point_index];
        auto * features = point_embedding_features.data() + point_index * point_input_width;
        for (std::size_t projection = 0; projection < projection_width; ++projection) {
            const float value = point.x * basis_values[projection] +
                                point.y * basis_values[projection_width + projection] +
                                point.z * basis_values[2U * projection_width + projection];
            features[projection] = std::sin(value);
            features[projection_width + projection] = std::cos(value);
        }
        features[2U * projection_width] = point.x;
        features[2U * projection_width + 1U] = point.y;
        features[2U * projection_width + 2U] = point.z;
    }

    auto * input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, static_cast<std::int64_t>(point_input_width), count);
    auto * trajectory_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 1008, count);
    auto * mask = ggml_new_tensor_4d(context, GGML_TYPE_F16, count, count, 1, 1);
    ggml_set_input(input); ggml_set_input(trajectory_input); ggml_set_input(mask);

    // GLSL sin/cos are only required to meet a loose accuracy bound, and the
    // Vulkan implementation is visibly inaccurate at this model's high
    // Fourier frequencies. Computing this tiny, model-defined feature map on
    // the host prevents that error from being amplified by the VAE.
    auto * point_features = ops.linear(input, "dvae.point_embed.mlp");
    auto * point_initial = point_features;
    auto * trajectory = ops.linear(trajectory_input, "dvae.traj_embed.mlp");

    const std::string neighbor = "dvae.neighbor_layers.0";
    auto * qn = ops.layer_norm(point_features, neighbor + ".norm_q", 1.0e-5F);
    auto * kn = ops.layer_norm(point_features, neighbor + ".norm_k", 1.0e-5F);
    auto * vn = ops.layer_norm(point_features, neighbor + ".norm_v", 1.0e-5F);
    auto * q = ops.linear(qn, neighbor + ".fn.to_q");
    auto * k = ops.linear(kn, neighbor + ".fn.to_k");
    auto * v = ops.linear(vn, neighbor + ".fn.to_v");
    auto * attended = ops.attention(q, k, v, neighbor_inner, 1, count, count, 1, mask);
    auto * neighbor_residual = ops.linear(attended, neighbor + ".fn.to_out");
    point_features = point_features == nullptr || neighbor_residual == nullptr ? nullptr
        : ggml_add(context, point_features, neighbor_residual);

    if (!ops.missing().empty() || point_features == nullptr || trajectory == nullptr) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "missing VAE embedding tensor '" + ops.missing() + "'"));
    }
    point_features = ggml_cont(context, point_features);
    trajectory = ggml_cont(context, trajectory);
    ggml_set_output(point_features); ggml_set_output(trajectory);
    if (trace != nullptr) {
        point_initial = ggml_cont(context, point_initial);
        ggml_set_output(point_initial);
    }
    ggml_build_forward_expand(graph, point_features);
    ggml_build_forward_expand(graph, trajectory);
    if (trace != nullptr) ggml_build_forward_expand(graph, point_initial);

    detail::vae_embeddings result_values;
    result_values.points = points.size();
    result_values.point.resize(points.size() * static_cast<std::size_t>(width));
    result_values.trajectory.resize(points.size() * static_cast<std::size_t>(width));
    const std::array uploads{
        detail::upload{input, point_embedding_features.data(),
                       point_embedding_features.size() * sizeof(float)},
        detail::upload{trajectory_input, trajectory_features.data(), trajectory_features.size() * sizeof(float)},
        detail::upload{mask, mask_values.data(), mask_values.size() * sizeof(ggml_fp16_t)},
    };
    std::vector<detail::download> downloads{
        {point_features, result_values.point.data(), result_values.point.size() * sizeof(float)},
        {trajectory, result_values.trajectory.data(), result_values.trajectory.size() * sizeof(float)},
    };
    if (trace != nullptr) {
        trace->point_initial.resize(result_values.point.size());
        downloads.push_back({point_initial, trace->point_initial.data(),
                             trace->point_initial.size() * sizeof(float)});
    }
    auto status = detail::execute_graph(
        backend->value, compute_allocator, graph, uploads, downloads);
    if (!status) return std::unexpected(status.error());
    return result_values;
}

result<detail::vae_encoded> model::impl::vae_encode(
    const detail::vae_embeddings & embeddings,
    std::span<const std::uint32_t> indices,
    std::span<const float> epsilon,
    std::vector<std::vector<float>> * block_outputs) const {
    constexpr std::int64_t width = 256;
    constexpr std::int64_t latent_width = 32;
    if (embeddings.points == 0U || indices.empty() ||
        embeddings.point.size() != embeddings.points * static_cast<std::size_t>(width) ||
        embeddings.trajectory.size() != embeddings.point.size() ||
        epsilon.size() != indices.size() * static_cast<std::size_t>(latent_width) ||
        std::ranges::any_of(indices, [&](std::uint32_t value) { return value >= embeddings.points; })) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "invalid VAE encoder features"));
    }
    const auto point_count = static_cast<std::int64_t>(embeddings.points);
    const auto trajectory_count = static_cast<std::int64_t>(indices.size());
    std::vector<std::int32_t> signed_indices(indices.begin(), indices.end());

    std::scoped_lock lock{compute_mutex};
    detail::graph_arena arena{16384U};
    if (!arena.valid()) return std::unexpected(detail::fail(error_code::allocation, "failed to create VAE encoder graph"));
    auto * context = arena.context(); auto * graph = arena.graph();
    detail::graph_ops ops{context, tensors};
    auto * point_original = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, point_count);
    auto * trajectory_original = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, point_count);
    auto * index_tensor = ggml_new_tensor_1d(context, GGML_TYPE_I32, trajectory_count);
    auto * epsilon_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, latent_width, trajectory_count);
    ggml_set_input(point_original); ggml_set_input(trajectory_original);
    ggml_set_input(index_tensor); ggml_set_input(epsilon_tensor);
    auto * point = ggml_get_rows(context, point_original, index_tensor);
    auto * trajectory = ggml_get_rows(context, trajectory_original, index_tensor);
    std::vector<ggml_tensor *> trace_tensors;
    if (block_outputs != nullptr) trace_tensors.reserve(32U);

    for (std::int64_t block = 0; block < 8; ++block) {
        const std::string prefix = "dvae.enc_blocks." + std::to_string(block);
        auto [point_attention, trajectory_attention] = synchronized_attention(
            context, ops, point, point_original, point_original, trajectory_original,
            prefix + ".0", width, width, 1, trajectory_count, point_count, 1);
        point = point_attention == nullptr ? nullptr : ggml_add(context, point, point_attention);
        trajectory = trajectory_attention == nullptr ? nullptr
                                                      : ggml_add(context, trajectory, trajectory_attention);
        if (block_outputs != nullptr) {
            trace_tensors.push_back(point);
            trace_tensors.push_back(trajectory);
        }
        auto [point_ffn, trajectory_ffn] = synchronized_feed_forward(
            context, ops, point, trajectory, prefix + ".1", width, trajectory_count, 1);
        point = point_ffn == nullptr ? nullptr : ggml_add(context, point, point_ffn);
        trajectory = trajectory_ffn == nullptr ? nullptr : ggml_add(context, trajectory, trajectory_ffn);
        if (block_outputs != nullptr) {
            trace_tensors.push_back(point);
            trace_tensors.push_back(trajectory);
        }
    }
    auto * x0 = ops.linear(point, "dvae.mean_fc_x0");
    auto * mean = ops.linear(trajectory, "dvae.mean_fc_xt");
    auto * log_variance = ops.linear(trajectory, "dvae.logvar_fc_xt");
    if (log_variance != nullptr) log_variance = ggml_clamp(context, log_variance, -30.0F, 20.0F);
    auto * standard_deviation = log_variance == nullptr ? nullptr
        : ggml_exp(context, ggml_scale(context, log_variance, 0.5F));
    auto * xt = mean == nullptr || standard_deviation == nullptr ? nullptr
        : ggml_add(context, mean, ggml_mul(context, standard_deviation, epsilon_tensor));
    auto * latent = x0 == nullptr || xt == nullptr ? nullptr : ggml_concat(context, x0, xt, 0);
    if (!ops.missing().empty() || latent == nullptr) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "missing VAE encoder tensor '" + ops.missing() + "'"));
    }
    latent = ggml_cont(context, latent);
    std::vector<std::vector<float>> trace_values;
    std::vector<detail::download> downloads;
    if (block_outputs != nullptr) {
        trace_values.resize(trace_tensors.size());
        downloads.reserve(trace_tensors.size() + 1U);
        for (std::size_t index = 0; index < trace_tensors.size(); ++index) {
            auto * tensor = ggml_cont(context, trace_tensors[index]);
            trace_tensors[index] = tensor;
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
            trace_values[index].resize(indices.size() * static_cast<std::size_t>(width));
            downloads.push_back({tensor, trace_values[index].data(),
                                 trace_values[index].size() * sizeof(float)});
        }
    }
    ggml_set_output(latent); ggml_build_forward_expand(graph, latent);
    detail::vae_encoded result_values;
    result_values.trajectories = indices.size();
    result_values.latent.resize(indices.size() * 64U);
    const std::array uploads{
        detail::upload{point_original, embeddings.point.data(), embeddings.point.size() * sizeof(float)},
        detail::upload{trajectory_original, embeddings.trajectory.data(), embeddings.trajectory.size() * sizeof(float)},
        detail::upload{index_tensor, signed_indices.data(), signed_indices.size() * sizeof(std::int32_t)},
        detail::upload{epsilon_tensor, epsilon.data(), epsilon.size_bytes()},
    };
    downloads.push_back(
        {latent, result_values.latent.data(), result_values.latent.size() * sizeof(float)});
    auto status = detail::execute_graph(
        backend->value, compute_allocator, graph, uploads, downloads);
    if (!status) return std::unexpected(status.error());
    if (block_outputs != nullptr) *block_outputs = std::move(trace_values);
    return result_values;
}

result<std::vector<float>> model::impl::vae_decode_offsets(
    const detail::vae_embeddings & embeddings,
    std::span<const float> latent, std::size_t trajectories,
    std::vector<std::vector<float>> * block_outputs) const {
    constexpr std::int64_t width = 256;
    constexpr std::int64_t inner = 512;
    if (embeddings.points == 0U || trajectories == 0U ||
        latent.size() != trajectories * 64U ||
        embeddings.point.size() != embeddings.points * static_cast<std::size_t>(width)) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "invalid VAE decoder features"));
    }
    const auto point_count = static_cast<std::int64_t>(embeddings.points);
    const auto trajectory_count = static_cast<std::int64_t>(trajectories);
    std::scoped_lock lock{compute_mutex};
    detail::graph_arena arena{16384U};
    if (!arena.valid()) return std::unexpected(detail::fail(error_code::allocation, "failed to create VAE decoder graph"));
    auto * context = arena.context(); auto * graph = arena.graph();
    detail::graph_ops ops{context, tensors};
    auto * latent_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, 64, trajectory_count);
    auto * point_original = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, point_count);
    ggml_set_input(latent_tensor); ggml_set_input(point_original);
    auto * latent_x0 = ggml_cont(context, ggml_view_2d(
        context, latent_tensor, 32, trajectory_count, latent_tensor->nb[1], 0));
    auto * latent_xt = ggml_cont(context, ggml_view_2d(
        context, latent_tensor, 32, trajectory_count, latent_tensor->nb[1],
        32U * latent_tensor->nb[0]));
    auto * x0 = ops.linear(latent_x0, "dvae.proj_x0");
    auto * xt = ops.linear(latent_xt, "dvae.proj_xt");
    std::vector<ggml_tensor *> trace_tensors;
    if (block_outputs != nullptr) trace_tensors.reserve(32U);
    for (std::int64_t block = 0; block < 8; ++block) {
        const std::string prefix = "dvae.dec_blocks." + std::to_string(block);
        auto [x0_attention, xt_attention] = synchronized_attention(
            context, ops, x0, x0, x0, xt, prefix + ".0",
            width, inner, 8, trajectory_count, trajectory_count, 1);
        x0 = x0_attention == nullptr ? nullptr : ggml_add(context, x0, x0_attention);
        xt = xt_attention == nullptr ? nullptr : ggml_add(context, xt, xt_attention);
        if (block_outputs != nullptr) {
            trace_tensors.push_back(x0);
            trace_tensors.push_back(xt);
        }
        auto [x0_ffn, xt_ffn] = synchronized_feed_forward(
            context, ops, x0, xt, prefix + ".1", width, trajectory_count, 1);
        x0 = x0_ffn == nullptr ? nullptr : ggml_add(context, x0, x0_ffn);
        xt = xt_ffn == nullptr ? nullptr : ggml_add(context, xt, xt_ffn);
        if (block_outputs != nullptr) {
            trace_tensors.push_back(x0);
            trace_tensors.push_back(xt);
        }
    }
    auto * query = ops.linear(point_original, "dvae.fc_query");
    const std::string final_prefix = "dvae.decoder_final_ca";
    auto * qn = ops.layer_norm(query, final_prefix + ".norm_q", 1.0e-5F);
    auto * kn = ops.layer_norm(x0, final_prefix + ".norm_k", 1.0e-5F);
    auto * vn = ops.layer_norm(xt, final_prefix + ".norm_v", 1.0e-5F);
    auto * q = ops.linear(qn, final_prefix + ".fn.to_q");
    auto * k = ops.linear(kn, final_prefix + ".fn.to_k");
    auto * v = ops.linear(vn, final_prefix + ".fn.to_v");
    auto * attended = ops.attention(q, k, v, inner, 1, point_count, trajectory_count, 1);
    auto * cross_attention = ops.linear(attended, final_prefix + ".fn.to_out");
    auto * offsets = ops.linear(cross_attention, "dvae.to_outputs");
    if (!ops.missing().empty() || offsets == nullptr) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "missing VAE decoder tensor '" + ops.missing() + "'"));
    }
    offsets = ggml_cont(context, offsets);
    std::vector<std::vector<float>> trace_values;
    std::vector<detail::download> downloads;
    if (block_outputs != nullptr) {
        trace_values.resize(trace_tensors.size());
        downloads.reserve(trace_tensors.size() + 1U);
        for (std::size_t index = 0; index < trace_tensors.size(); ++index) {
            auto * tensor = ggml_cont(context, trace_tensors[index]);
            trace_tensors[index] = tensor;
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
            trace_values[index].resize(trajectories * static_cast<std::size_t>(width));
            downloads.push_back({tensor, trace_values[index].data(),
                                 trace_values[index].size() * sizeof(float)});
        }
    }
    ggml_set_output(offsets); ggml_build_forward_expand(graph, offsets);
    std::vector<float> result_values(embeddings.points * 48U);
    const std::array uploads{
        detail::upload{latent_tensor, latent.data(), latent.size_bytes()},
        detail::upload{point_original, embeddings.point.data(), embeddings.point.size() * sizeof(float)},
    };
    downloads.push_back({offsets, result_values.data(), result_values.size() * sizeof(float)});
    auto status = detail::execute_graph(
        backend->value, compute_allocator, graph, uploads, downloads);
    if (!status) return std::unexpected(status.error());
    if (block_outputs != nullptr) *block_outputs = std::move(trace_values);
    return result_values;
}

} // namespace aam
