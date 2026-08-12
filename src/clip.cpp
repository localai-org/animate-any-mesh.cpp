#include "graph.hpp"

#include <algorithm>
#include <limits>

namespace aam {

result<std::vector<float>> model::impl::clip_encode(
    std::span<const std::int32_t> tokens, std::size_t batch,
    std::vector<std::vector<float>> * block_outputs) const {
    constexpr std::int64_t context_length = 77;
    constexpr std::int64_t width = 768;
    constexpr std::int64_t heads = 12;
    constexpr std::int64_t layers = 12;

    if (batch == 0U || batch > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        tokens.size() != batch * static_cast<std::size_t>(context_length)) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "CLIP token tensor must have shape [batch, 77]"));
    }
    std::scoped_lock lock{compute_mutex};
    detail::graph_arena arena{8192U};
    if (!arena.valid()) {
        return std::unexpected(detail::fail(error_code::allocation,
            "failed to create CLIP graph metadata arena"));
    }
    auto * context = arena.context();
    auto * graph = arena.graph();
    detail::graph_ops ops{context, tensors};

    // ggml_get_rows interprets a 2-D index tensor as a batched lookup and
    // requires a correspondingly batched embedding table. Flatten the IDs,
    // then restore [width, context, batch], so one immutable table serves all
    // classifier-free-guidance batches.
    auto * token_ids = ggml_new_tensor_1d(
        context, GGML_TYPE_I32, context_length * static_cast<std::int64_t>(batch));
    ggml_set_input(token_ids);
    ggml_set_name(token_ids, "clip.token_ids");
    auto * token_embedding = ops.weight(
        "rf.clip_text_model.text_model.embeddings.token_embedding.weight");
    auto * position_embedding = ops.weight(
        "rf.clip_text_model.text_model.embeddings.position_embedding.weight");
    auto * hidden = token_embedding == nullptr ? nullptr : ggml_get_rows(context, token_embedding, token_ids);
    if (hidden != nullptr) {
        hidden = ggml_reshape_3d(
            context, hidden, width, context_length, static_cast<std::int64_t>(batch));
        if (hidden->type != GGML_TYPE_F32) hidden = ggml_cast(context, hidden, GGML_TYPE_F32);
    }
    if (hidden != nullptr && position_embedding != nullptr) {
        auto * positions = position_embedding->type == GGML_TYPE_F32
            ? position_embedding : ggml_cast(context, position_embedding, GGML_TYPE_F32);
        hidden = ggml_add(context, hidden, positions);
    }

    std::vector<ggml_tensor *> captured_blocks;
    auto capture = [&](ggml_tensor * tensor) {
        if (block_outputs == nullptr || tensor == nullptr) return;
        auto * contiguous = ggml_cont(context, tensor);
        ggml_set_output(contiguous);
        ggml_build_forward_expand(graph, contiguous);
        captured_blocks.push_back(contiguous);
    };

    auto * causal_mask = ggml_new_tensor_4d(context, GGML_TYPE_F16,
        context_length, context_length, 1, 1);
    ggml_set_input(causal_mask);
    ggml_set_name(causal_mask, "clip.causal_mask");

    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "rf.clip_text_model.text_model.encoder.layers." +
                                   std::to_string(layer);
        auto * residual = hidden;
        auto * normalized = ops.layer_norm(hidden, prefix + ".layer_norm1", 1.0e-5F);
        auto * q = ops.linear(normalized, prefix + ".self_attn.q_proj");
        auto * k = ops.linear(normalized, prefix + ".self_attn.k_proj");
        auto * v = ops.linear(normalized, prefix + ".self_attn.v_proj");
        auto * attended = ops.attention(q, k, v, width, heads, context_length,
                                       context_length, static_cast<std::int64_t>(batch), causal_mask);
        attended = ops.linear(attended, prefix + ".self_attn.out_proj");
        hidden = residual == nullptr || attended == nullptr ? nullptr : ggml_add(context, residual, attended);

        residual = hidden;
        normalized = ops.layer_norm(hidden, prefix + ".layer_norm2", 1.0e-5F);
        auto * feed_forward = ops.linear(normalized, prefix + ".mlp.fc1");
        if (feed_forward != nullptr) feed_forward = ggml_gelu_quick(context, feed_forward);
        feed_forward = ops.linear(feed_forward, prefix + ".mlp.fc2");
        hidden = residual == nullptr || feed_forward == nullptr ? nullptr
                                                                : ggml_add(context, residual, feed_forward);
        capture(hidden);
    }
    hidden = ops.layer_norm(hidden,
        "rf.clip_text_model.text_model.final_layer_norm", 1.0e-5F);
    if (!ops.missing().empty() || hidden == nullptr) {
        return std::unexpected(detail::fail(error_code::incompatible_model,
            "missing CLIP tensor '" + ops.missing() + "'"));
    }
    hidden = ggml_cont(context, hidden);
    ggml_set_output(hidden);
    ggml_build_forward_expand(graph, hidden);

    std::vector<ggml_fp16_t> mask(static_cast<std::size_t>(context_length * context_length),
                                  ggml_fp32_to_fp16(0.0F));
    for (std::int64_t query = 0; query < context_length; ++query) {
        for (std::int64_t key = query + 1; key < context_length; ++key) {
            mask[static_cast<std::size_t>(key + context_length * query)] =
                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        }
    }
    const std::array uploads{
        detail::upload{token_ids, tokens.data(), tokens.size_bytes()},
        detail::upload{causal_mask, mask.data(), mask.size() * sizeof(ggml_fp16_t)},
    };
    std::vector<float> result_values(static_cast<std::size_t>(width * context_length) * batch);
    std::vector<std::vector<float>> captured_values(captured_blocks.size());
    std::vector<detail::download> downloads;
    downloads.reserve(captured_blocks.size() + 1U);
    for (std::size_t i = 0; i < captured_blocks.size(); ++i) {
        captured_values[i].resize(static_cast<std::size_t>(ggml_nelements(captured_blocks[i])));
        downloads.push_back({captured_blocks[i], captured_values[i].data(),
                             captured_values[i].size() * sizeof(float)});
    }
    downloads.push_back({hidden, result_values.data(), result_values.size() * sizeof(float)});
    auto status = detail::execute_graph(
        backend->value, compute_allocator, graph, uploads, downloads);
    if (!status) return std::unexpected(status.error());

    if (block_outputs != nullptr) *block_outputs = std::move(captured_values);

    return result_values;
}

} // namespace aam
