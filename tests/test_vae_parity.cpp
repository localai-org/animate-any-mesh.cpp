#include "model_impl.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace aam {
struct validation_access {
    static model::impl * get(model & value) { return value.impl_.get(); }
};
} // namespace aam

namespace {

struct gguf_deleter { void operator()(gguf_context * value) const { gguf_free(value); } };
struct ggml_deleter { void operator()(ggml_context * value) const { ggml_free(value); } };

template<class T>
std::span<const T> tensor_span(ggml_context * context, const char * name, ggml_type type) {
    auto * tensor = ggml_get_tensor(context, name);
    if (tensor == nullptr || tensor->type != type) return {};
    return {static_cast<const T *>(tensor->data), static_cast<std::size_t>(ggml_nelements(tensor))};
}

bool compare(std::string_view name, std::span<const float> got, std::span<const float> expected,
             float atol, float rtol) {
    if (got.size() != expected.size() || got.empty()) {
        std::cerr << name << ": size mismatch\n";
        return false;
    }
    float maximum = 0.0F;
    std::size_t worst = 0;
    std::size_t failures = 0;
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float difference = std::abs(got[i] - expected[i]);
        if (difference > maximum) { maximum = difference; worst = i; }
        if (difference > atol + rtol * std::abs(expected[i])) ++failures;
        numerator += static_cast<double>(difference) * difference;
        denominator += static_cast<double>(expected[i]) * expected[i];
    }
    std::cout << name << ": max_abs=" << maximum
              << " rel_l2=" << std::sqrt(numerator / denominator)
              << " worst=" << worst << " got=" << got[worst]
              << " expected=" << expected[worst] << " failures=" << failures << '\n';
    return failures == 0U;
}

} // namespace

int main() {
    const char * model_path = std::getenv("AAM_TEST_GGUF");
    const char * reference_path = std::getenv("AAM_VAE_REFERENCE");
    if (model_path == nullptr || reference_path == nullptr) {
        std::cout << "SKIP: set AAM_TEST_GGUF and AAM_VAE_REFERENCE\n";
        return 77;
    }
    ggml_context * raw_context = nullptr;
    gguf_init_params parameters{};
    parameters.no_alloc = false;
    parameters.ctx = &raw_context;
    std::unique_ptr<gguf_context, gguf_deleter> reference(
        gguf_init_from_file(reference_path, parameters));
    std::unique_ptr<ggml_context, ggml_deleter> context(raw_context);
    if (!reference || !context) return 1;

    const auto point_values = tensor_span<float>(context.get(), "input.pc0", GGML_TYPE_F32);
    const auto index_values = tensor_span<std::int32_t>(context.get(), "enc.indices", GGML_TYPE_I32);
    const auto epsilon = tensor_span<float>(context.get(), "enc.latent.epsilon", GGML_TYPE_F32);
    if (point_values.size() % 3U != 0U || index_values.empty() || epsilon.empty()) return 1;
    std::vector<aam::vec3> points(point_values.size() / 3U);
    static_assert(sizeof(aam::vec3) == 3U * sizeof(float));
    std::memcpy(points.data(), point_values.data(), point_values.size_bytes());
    std::vector<std::uint32_t> indices;
    indices.reserve(index_values.size());
    for (const auto value : index_values) {
        if (value < 0) return 1;
        indices.push_back(static_cast<std::uint32_t>(value));
    }
    std::vector<std::uint32_t> offsets(points.size() + 1U);
    std::vector<std::uint32_t> adjacency;
    adjacency.reserve(points.size() * 2U);
    for (std::size_t i = 0; i < points.size(); ++i) {
        offsets[i] = static_cast<std::uint32_t>(adjacency.size());
        adjacency.push_back(static_cast<std::uint32_t>((i + points.size() - 1U) % points.size()));
        adjacency.push_back(static_cast<std::uint32_t>((i + 1U) % points.size()));
    }
    offsets.back() = static_cast<std::uint32_t>(adjacency.size());

    aam::runtime_options options;
    options.device = std::getenv("AAM_PARITY_VULKAN") == nullptr
        ? aam::device_kind::cpu : aam::device_kind::vulkan;
    auto loaded = aam::model::load(model_path, options);
    if (!loaded) { std::cerr << loaded.error().message << '\n'; return 1; }
    auto * implementation = aam::validation_access::get(*loaded);
    aam::detail::vae_embedding_trace embedding_trace;
    auto embeddings = implementation->vae_embed(points, offsets, adjacency, &embedding_trace);
    if (!embeddings) { std::cerr << embeddings.error().message << '\n'; return 1; }
    bool okay = compare("vae.embed.point_initial", embedding_trace.point_initial,
        tensor_span<float>(context.get(), "enc.point", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F);
    okay = compare("vae.embed.point", embeddings->point,
        tensor_span<float>(context.get(), "enc.neighbor.output", GGML_TYPE_F32), 2.0e-3F, 2.0e-4F);
    okay = compare("vae.embed.trajectory", embeddings->trajectory,
        tensor_span<float>(context.get(), "enc.trajectory", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F) && okay;

    std::vector<std::vector<float>> encoder_blocks;
    auto encoded = implementation->vae_encode(*embeddings, indices, epsilon, &encoder_blocks);
    if (!encoded) { std::cerr << encoded.error().message << '\n'; return 1; }
    if (encoder_blocks.size() != 32U) return 1;
    for (std::size_t block = 0; block < 8U; ++block) {
        const auto prefix = "enc.block." + std::to_string(block);
        const std::array names{
            prefix + ".attention.point", prefix + ".attention.trajectory",
            prefix + ".point", prefix + ".trajectory"};
        for (std::size_t stage = 0; stage < names.size(); ++stage) {
            okay = compare(names[stage], encoder_blocks[block * 4U + stage],
                tensor_span<float>(context.get(), names[stage].c_str(), GGML_TYPE_F32),
                2.0e-4F, 2.0e-4F) && okay;
        }
    }
    okay = compare("vae.encoder.latent", encoded->latent,
        tensor_span<float>(context.get(), "enc.latent", GGML_TYPE_F32), 8.0e-3F, 5.0e-4F) && okay;

    std::vector<std::vector<float>> decoder_blocks;
    const auto reference_latent = tensor_span<float>(context.get(), "enc.latent", GGML_TYPE_F32);
    auto decoded = implementation->vae_decode_offsets(
        *embeddings, reference_latent, encoded->trajectories, &decoder_blocks);
    if (!decoded) { std::cerr << decoded.error().message << '\n'; return 1; }
    if (decoder_blocks.size() != 32U) return 1;
    for (std::size_t block = 0; block < 8U; ++block) {
        const auto prefix = "dec.block." + std::to_string(block);
        const std::array names{
            prefix + ".attention.x0", prefix + ".attention.xt",
            prefix + ".x0", prefix + ".xt"};
        for (std::size_t stage = 0; stage < names.size(); ++stage) {
            okay = compare(names[stage], decoder_blocks[block * 4U + stage],
                tensor_span<float>(context.get(), names[stage].c_str(), GGML_TYPE_F32),
                1.0e-3F, 4.0e-4F) && okay;
        }
    }
    okay = compare("vae.decoder.offsets", *decoded,
        tensor_span<float>(context.get(), "dec.offsets", GGML_TYPE_F32), 1.0e-2F, 8.0e-4F) && okay;

    std::vector<float> output(points.size() * 16U * 3U);
    for (std::size_t frame = 0; frame < 16U; ++frame) {
        for (std::size_t point = 0; point < points.size(); ++point) {
            const std::size_t output_index = (frame * points.size() + point) * 3U;
            const std::size_t offset_index = point * 48U + frame * 3U;
            output[output_index] = points[point].x + (*decoded)[offset_index];
            output[output_index + 1U] = points[point].y + (*decoded)[offset_index + 1U];
            output[output_index + 2U] = points[point].z + (*decoded)[offset_index + 2U];
        }
    }
    okay = compare("vae.output", output,
        tensor_span<float>(context.get(), "output", GGML_TYPE_F32), 1.0e-2F, 8.0e-4F) && okay;
    return okay ? 0 : 1;
}
