#include "graph.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace aam::detail {
namespace {

class graph_profiler final {
public:
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::duration<double, std::milli>;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    void record(duration allocation, duration upload, duration compute, duration download) {
        if (!enabled_) return;
        std::scoped_lock lock{mutex_};
        ++calls_;
        allocation_ += allocation;
        upload_ += upload;
        compute_ += compute;
        download_ += download;
    }

    ~graph_profiler() {
        if (!enabled_) return;
        const double total = allocation_.count() + upload_.count() +
                             compute_.count() + download_.count();
        std::fprintf(stderr,
            "aam_profile graph_exec calls=%zu allocation_ms=%.3f upload_ms=%.3f "
            "compute_submit_ms=%.3f download_sync_ms=%.3f total_ms=%.3f\n",
            calls_, allocation_.count(), upload_.count(), compute_.count(),
            download_.count(), total);
    }

private:
    bool enabled_ = std::getenv("AAM_PROFILE") != nullptr;
    std::mutex mutex_;
    std::size_t calls_ = 0U;
    duration allocation_{};
    duration upload_{};
    duration compute_{};
    duration download_{};
};

graph_profiler & profiler() {
    static graph_profiler instance;
    return instance;
}

void replace_all(std::string & value, std::string_view from, std::string_view to) {
    for (std::size_t position = 0; (position = value.find(from, position)) != std::string::npos;) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

std::string compact_name(std::string name) {
    std::string result;
    if (name.starts_with("rf.clip_text_model.text_model.")) {
        result = "c." + name.substr(std::string_view{"rf.clip_text_model.text_model."}.size());
    } else if (name.starts_with("rf.backbone.resblocks.")) {
        result = "r.b." + name.substr(std::string_view{"rf.backbone.resblocks."}.size());
    } else if (name.starts_with("dvae.")) {
        result = "v." + name.substr(5U);
    } else if (name.starts_with("rf.")) {
        result = "r." + name.substr(3U);
    } else {
        result = std::move(name);
    }
    constexpr std::pair<std::string_view, std::string_view> replacements[] = {
        {"encoder.layers.", "l."}, {"embeddings.", "em."},
        {"token_embedding", "tok"}, {"position_embedding", "pos"},
        {"final_layer_norm", "fn"}, {"neighbor_layers", "nl"},
        {"enc_blocks", "eb"}, {"dec_blocks", "db"},
        {"point_embed", "pe"}, {"traj_embed", "te"},
        {"decoder_final_ca", "ca"}, {"mean_fc_x0", "mx0"},
        {"mean_fc_xt", "mxt"}, {"logvar_fc_xt", "lvxt"},
        {"proj_x0", "px0"}, {"proj_xt", "pxt"}, {"fc_query", "fq"},
        {"to_outputs", "out"}, {"time_embed", "t"},
        {"clip_token_mlp", "ct"}, {"input_proj", "in"}, {"output_proj", "out"},
        {"self_attn", "sa"}, {"layer_norm1", "n1"}, {"layer_norm2", "n2"},
        {"ln_pre_text", "npt"}, {"ln_pre", "np"}, {"ln_post", "no"},
        {"adaln_text", "at"}, {"adaln_x", "ax"},
        {"norm_q", "nq"}, {"norm_k", "nk"}, {"norm_v1", "nv1"},
        {"norm_v2", "nv2"}, {"norm_v", "nv"}, {"freq_bands", "freq"},
        {".attention.", ".a."}, {".attn.", ".a."}, {".linear.", ".l."},
        {".weight", ".w"}, {".bias", ".b"}, {".to_out1", ".o1"},
        {".to_out2", ".o2"}, {".to_out", ".o"}, {".to_qkv", ".qkv"},
        {".to_q", ".q"}, {".to_k", ".k"}, {".to_v1", ".v1"},
        {".to_v2", ".v2"}, {".to_v", ".v"}, {".c_qkv", ".qkv"},
        {".c_proj", ".p"}, {".c_fc", ".f"}, {".ffn.", ".f."},
        {".mlp.", ".m."},
    };
    for (const auto & [from, to] : replacements) replace_all(result, from, to);
    return result;
}

} // namespace

graph_arena::graph_arena(std::size_t capacity) {
    const auto metadata_bytes = ggml_tensor_overhead() * capacity +
                                ggml_graph_overhead_custom(capacity, false);
    ggml_init_params parameters{metadata_bytes, nullptr, true};
    context_ = ggml_init(parameters);
    if (context_ != nullptr) {
        graph_ = ggml_new_graph_custom(context_, capacity, false);
    }
}

graph_arena::~graph_arena() {
    if (context_ != nullptr) ggml_free(context_);
}

ggml_tensor * graph_ops::weight(const std::string & name) {
    const auto compact = compact_name(name);
    const auto it = tensors_.find(compact);
    if (it == tensors_.end()) {
        if (missing_.empty()) missing_ = name;
        return nullptr;
    }
    return it->second;
}

ggml_tensor * graph_ops::optional_weight(const std::string & name) const {
    const auto it = tensors_.find(compact_name(name));
    return it == tensors_.end() ? nullptr : it->second;
}

ggml_tensor * graph_ops::linear(ggml_tensor * input, const std::string & prefix) {
    auto * matrix = weight(prefix + ".weight");
    if (matrix == nullptr || input == nullptr) return nullptr;
    auto * output = ggml_mul_mat(context_, matrix, input);
    if (auto * bias = optional_weight(prefix + ".bias"); bias != nullptr) {
        output = ggml_add(context_, output, bias);
    }
    return output;
}

ggml_tensor * graph_ops::layer_norm(
    ggml_tensor * input, const std::string & prefix, float epsilon) {
    auto * scale = weight(prefix + ".weight");
    auto * shift = weight(prefix + ".bias");
    if (input == nullptr || scale == nullptr || shift == nullptr) return nullptr;
    auto * output = ggml_norm(context_, input, epsilon);
    output = ggml_mul(context_, output, scale);
    return ggml_add(context_, output, shift);
}

ggml_tensor * graph_ops::attention(
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    std::int64_t width, std::int64_t heads,
    std::int64_t query_tokens, std::int64_t key_tokens,
    std::int64_t batch, ggml_tensor * mask) {
    return attention_values(q, k, v, width, width, heads, query_tokens,
                            key_tokens, batch, mask);
}

ggml_tensor * graph_ops::attention_values(
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    std::int64_t key_width, std::int64_t value_width, std::int64_t heads,
    std::int64_t query_tokens, std::int64_t key_tokens,
    std::int64_t batch, ggml_tensor * mask) {
    if (q == nullptr || k == nullptr || v == nullptr || heads <= 0 ||
        key_width % heads != 0 || value_width % heads != 0) {
        return nullptr;
    }
    const std::int64_t key_head_width = key_width / heads;
    const std::int64_t value_head_width = value_width / heads;
    auto reshape = [&](ggml_tensor * tensor, std::int64_t tensor_width, std::int64_t tokens) {
        auto * shaped = ggml_reshape_4d(context_, tensor, tensor_width / heads, heads, tokens, batch);
        return ggml_cont(context_, ggml_permute(context_, shaped, 0, 2, 1, 3));
    };
    auto * query = reshape(q, key_width, query_tokens); // [Dk, Nq, H, B]
    auto * key = reshape(k, key_width, key_tokens);     // [Dk, Nk, H, B]
    auto * value = reshape(v, value_width, key_tokens); // [Dv, Nk, H, B]
    const float scale = 1.0F / std::sqrt(static_cast<float>(key_head_width));

    const bool exact = std::getenv("AAM_EXACT_ATTENTION") != nullptr;
    ggml_tensor * output = nullptr;
    if (!exact) {
        output = ggml_flash_attn_ext(context_, query, key, value, mask, scale, 0.0F, 0.0F);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    } else {
        auto * scores = ggml_mul_mat(context_, key, query); // [Nk,Nq,H,B]
        if (mask != nullptr) {
            auto * additive_mask = mask->type == GGML_TYPE_F32
                ? mask : ggml_cast(context_, mask, GGML_TYPE_F32);
            scores = ggml_add(context_, scores, additive_mask);
        }
        scores = ggml_soft_max_ext(context_, scores, nullptr, scale, 0.0F);
        auto * transposed_value = ggml_cont(context_, ggml_permute(context_, value, 1, 0, 2, 3));
        output = ggml_mul_mat(context_, transposed_value, scores); // [D,Nq,H,B]
        output = ggml_cont(context_, ggml_permute(context_, output, 0, 2, 1, 3));
    }
    (void)value_head_width;
    return ggml_reshape_3d(context_, output, value_width, query_tokens, batch);
}

result<void> execute_graph(
    ggml_backend_t backend, ggml_gallocr_t allocator,
    ggml_cgraph * graph, std::span<const upload> uploads,
    std::span<const download> downloads) {
    if (backend == nullptr || allocator == nullptr || graph == nullptr) {
        return std::unexpected(fail(error_code::invalid_argument, "cannot execute a null ggml graph"));
    }
    auto & graph_profile = profiler();
    const auto allocation_started = graph_profiler::clock::now();
    if (!ggml_gallocr_alloc_graph(allocator, graph)) {
        return std::unexpected(fail(error_code::allocation, "failed to allocate ggml compute graph"));
    }
    const auto upload_started = graph_profiler::clock::now();
    for (const auto & item : uploads) {
        if (item.tensor == nullptr || item.data == nullptr || item.bytes != ggml_nbytes(item.tensor)) {
            return std::unexpected(fail(error_code::invalid_argument, "invalid ggml graph input upload"));
        }
        ggml_backend_tensor_set(item.tensor, item.data, 0, item.bytes);
    }
    const auto compute_started = graph_profiler::clock::now();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return std::unexpected(fail(error_code::compute, "ggml backend graph compute failed"));
    }
    const auto download_started = graph_profiler::clock::now();
    for (const auto & item : downloads) {
        if (item.tensor == nullptr || item.data == nullptr || item.bytes != ggml_nbytes(item.tensor)) {
            return std::unexpected(fail(error_code::invalid_argument, "invalid ggml graph output download"));
        }
        ggml_backend_tensor_get(item.tensor, item.data, 0, item.bytes);
    }
    const auto finished = graph_profiler::clock::now();
    if (graph_profile.enabled()) {
        graph_profile.record(upload_started - allocation_started,
                             compute_started - upload_started,
                             download_started - compute_started,
                             finished - download_started);
    }
    return {};
}

} // namespace aam::detail
