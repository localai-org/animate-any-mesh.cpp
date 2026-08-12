#include "model_impl.hpp"
#include "pytorch_rng.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
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
    return {static_cast<const T *>(tensor->data),
            static_cast<std::size_t>(ggml_nelements(tensor))};
}

bool compare(const char * name, std::span<const float> got, std::span<const float> expected) {
    if (got.size() != expected.size() || got.empty()) {
        std::cerr << name << ": size mismatch\n";
        return false;
    }
    float maximum = 0.0F;
    std::size_t worst = 0U;
    std::size_t failures = 0U;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0; index < got.size(); ++index) {
        const float difference = std::abs(got[index] - expected[index]);
        if (difference > maximum) {
            maximum = difference;
            worst = index;
        }
        if (difference > 2.0e-3F + 2.0e-3F * std::abs(expected[index])) ++failures;
        squared_error += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    const double relative = std::sqrt(squared_error / squared_reference);
    std::cout << name << ": max_abs=" << maximum << " rel_l2=" << relative
              << " worst=" << worst << " got=" << got[worst]
              << " expected=" << expected[worst] << " failures=" << failures << '\n';
    return failures == 0U;
}

void compare_motion_frames(
    std::span<const float> got, std::span<const float> expected,
    std::size_t vertices, std::size_t frames) {
    const std::size_t frame_values = vertices * 3U;
    if (frames < 2U || got.size() != frame_values * frames ||
        expected.size() != got.size()) return;
    for (std::size_t frame = 1U; frame < frames; ++frame) {
        double got_squared = 0.0;
        double expected_squared = 0.0;
        double difference_squared = 0.0;
        double dot = 0.0;
        for (std::size_t index = 0U; index < frame_values; ++index) {
            const double got_delta = static_cast<double>(
                got[frame * frame_values + index] - got[index]);
            const double expected_delta = static_cast<double>(
                expected[frame * frame_values + index] - expected[index]);
            const double difference = got_delta - expected_delta;
            got_squared += got_delta * got_delta;
            expected_squared += expected_delta * expected_delta;
            difference_squared += difference * difference;
            dot += got_delta * expected_delta;
        }
        const double cosine = dot / std::sqrt(got_squared * expected_squared);
        const double magnitude_ratio = std::sqrt(got_squared / expected_squared);
        const double relative = std::sqrt(difference_squared / expected_squared);
        std::cout << "full.motion.frame." << frame
                  << ": displacement_rel_l2=" << relative
                  << " direction_cosine=" << cosine
                  << " magnitude_ratio=" << magnitude_ratio << '\n';
    }
}

bool save_comparison_glbs(
    const std::filesystem::path & prefix, const aam::mesh & topology,
    std::span<const float> cpp, std::span<const float> upstream,
    std::size_t vertices, std::size_t frames) {
    auto make_animation = [&](std::span<const float> positions) {
        aam::animation value;
        value.frames = frames;
        value.vertices_per_frame = vertices;
        value.positions.resize(frames * vertices);
        static_assert(sizeof(aam::vec3) == 3U * sizeof(float));
        std::memcpy(value.positions.data(), positions.data(), positions.size_bytes());
        return value;
    };
    const auto cpp_path = std::filesystem::path(prefix.string() + "-cpp.glb");
    const auto upstream_path = std::filesystem::path(prefix.string() + "-upstream.glb");
    auto cpp_status = aam::save_animation_glb_file(
        cpp_path, topology, make_animation(cpp), 10.0F);
    auto upstream_status = aam::save_animation_glb_file(
        upstream_path, topology, make_animation(upstream), 10.0F);
    if (!cpp_status || !upstream_status) {
        std::cerr << "failed to save full-pipeline comparison GLBs\n";
        return false;
    }
    std::cout << "saved " << cpp_path << " and " << upstream_path << '\n';
    return true;
}

} // namespace

int main() {
    const char * model_path = std::getenv("AAM_TEST_GGUF");
    const char * reference_path = std::getenv("AAM_FULL_REFERENCE");
    if (model_path == nullptr || reference_path == nullptr) {
        std::cout << "SKIP: set AAM_TEST_GGUF and AAM_FULL_REFERENCE\n";
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

    const auto vertex_values = tensor_span<float>(
        context.get(), "input.vertices", GGML_TYPE_F32);
    const auto face_values = tensor_span<std::int32_t>(
        context.get(), "input.faces", GGML_TYPE_I32);
    const auto index_values = tensor_span<std::int32_t>(
        context.get(), "sample.indices", GGML_TYPE_I32);
    const auto epsilon = tensor_span<float>(
        context.get(), "sample.epsilon", GGML_TYPE_F32);
    const auto noise = tensor_span<float>(
        context.get(), "sample.noise", GGML_TYPE_F32);
    const auto expected_encoded = tensor_span<float>(
        context.get(), "expected.encoded", GGML_TYPE_F32);
    const auto expected_rf = tensor_span<float>(
        context.get(), "expected.rf_normalized", GGML_TYPE_F32);
    const auto expected_latent = tensor_span<float>(
        context.get(), "expected.latent", GGML_TYPE_F32);
    const auto expected = tensor_span<float>(
        context.get(), "output.positions", GGML_TYPE_F32);
    if (vertex_values.empty() || vertex_values.size() % 3U != 0U ||
        face_values.empty() || face_values.size() % 3U != 0U ||
        index_values.empty() || epsilon.size() != index_values.size() * 32U ||
        noise.size() != index_values.size() * 64U ||
        expected_encoded.size() != noise.size() || expected_rf.size() != noise.size() ||
        expected_latent.size() != noise.size() ||
        expected.size() != vertex_values.size() * aam::default_frames) return 1;

    if (std::getenv("AAM_TEST_PYTORCH_RNG") != nullptr) {
        aam::detail::pytorch_cuda_generator generator{666U};
        std::vector<float> generated_noise(noise.size());
        for (std::size_t index = 0U; index < generated_noise.size(); ++index) {
            generated_noise[index] = generator.normal(index, generated_noise.size());
        }
        const bool rng_ok = compare("full.pytorch_rng.noise", generated_noise, noise);
        std::cout << "full.pytorch_rng.replacement_index="
                  << generator.replacement_index(0U, vertex_values.size() / 3U)
                  << " expected=" << index_values.front() << '\n';
        if (std::getenv("AAM_TEST_PYTORCH_RNG_ONLY") != nullptr) {
            return rng_ok ? 0 : 1;
        }
    }

    aam::mesh source;
    source.vertices.resize(vertex_values.size() / 3U);
    static_assert(sizeof(aam::vec3) == 3U * sizeof(float));
    std::memcpy(source.vertices.data(), vertex_values.data(), vertex_values.size_bytes());
    source.faces.reserve(face_values.size() / 3U);
    for (std::size_t offset = 0; offset < face_values.size(); offset += 3U) {
        aam::triangle face{};
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const auto value = face_values[offset + corner];
            if (value < 0 || static_cast<std::size_t>(value) >= source.vertices.size()) return 1;
            face[corner] = static_cast<std::uint32_t>(value);
        }
        source.faces.push_back(face);
    }
    auto prepared = aam::prepare_mesh(source, source.vertices.size());
    if (!prepared || prepared->value.vertices.size() != source.vertices.size()) return 1;

    std::vector<std::uint32_t> indices;
    indices.reserve(index_values.size());
    for (const auto value : index_values) {
        if (value < 0 || static_cast<std::size_t>(value) >= prepared->value.vertices.size()) return 1;
        indices.push_back(static_cast<std::uint32_t>(value));
    }

    aam::runtime_options options;
    options.device = std::getenv("AAM_FULL_VULKAN") == nullptr
        ? aam::device_kind::cpu : aam::device_kind::vulkan;
    if (const char * directory = std::getenv("AAM_BACKEND_DIR")) {
        options.backend_directory = directory;
    }
    auto loaded = aam::model::load(model_path, options);
    if (!loaded) {
        std::cerr << loaded.error().message << '\n';
        return 1;
    }
    auto * implementation = aam::validation_access::get(*loaded);
    auto embeddings = implementation->vae_embed(
        prepared->value.vertices, prepared->adjacency_offsets,
        prepared->adjacency_indices);
    if (!embeddings) return 1;
    if (std::getenv("AAM_TEST_PRODUCTION_INDICES") != nullptr) {
        auto sampled = aam::farthest_point_sample(
            embeddings->point, embeddings->points, 256U,
            std::min(indices.size(), embeddings->points));
        if (!sampled) return 1;
        std::vector<std::uint32_t> production(indices.size(), 0U);
        std::copy(sampled->begin(), sampled->end(), production.begin());
        aam::detail::pytorch_cuda_generator generator{666U};
        for (std::size_t position = 0U; position < production.size(); ++position) {
            if (production[position] == 0U) {
                production[position] = static_cast<std::uint32_t>(
                    generator.replacement_index(position, embeddings->points));
            }
        }
        std::size_t ordered_matches = 0U;
        std::size_t first_mismatch = production.size();
        std::unordered_set<std::uint32_t> expected_set;
        expected_set.reserve(indices.size());
        for (const auto value : indices) expected_set.insert(value);
        std::size_t overlap = 0U;
        for (std::size_t position = 0U; position < production.size(); ++position) {
            if (production[position] == indices[position]) {
                ++ordered_matches;
            } else if (first_mismatch == production.size()) {
                first_mismatch = position;
            }
            if (expected_set.contains(production[position])) ++overlap;
        }
        std::cout << "full.production_indices: ordered_matches=" << ordered_matches
                  << '/' << production.size() << " set_overlap=" << overlap
                  << '/' << production.size();
        if (first_mismatch != production.size()) {
            std::cout << " first_mismatch=" << first_mismatch
                      << " got=" << production[first_mismatch]
                      << " expected=" << indices[first_mismatch];
        }
        std::cout << '\n';
        if (std::getenv("AAM_USE_PRODUCTION_INDICES") != nullptr) {
            indices = std::move(production);
        }
        if (std::getenv("AAM_TEST_INDICES_ONLY") != nullptr) return 0;
    }
    auto encoded = implementation->vae_encode(*embeddings, indices, epsilon);
    if (!encoded) return 1;
    bool okay = compare("full.encoded", encoded->latent, expected_encoded);
    if (std::getenv("AAM_FULL_ENCODE_ONLY") != nullptr) return okay ? 0 : 1;

    constexpr float f0_mean = 0.205227330327034F;
    constexpr float f0_std = 9.064417839050293F;
    constexpr float ft_mean = 0.95684415102005F;
    constexpr float ft_std = 6.397393703460693F;
    std::vector<float> f0(indices.size() * 32U);
    for (std::size_t token = 0; token < indices.size(); ++token) {
        for (std::size_t channel = 0; channel < 32U; ++channel) {
            f0[token * 32U + channel] =
                (encoded->latent[token * 64U + channel] - f0_mean) / f0_std;
        }
    }
    const char * prompt_text = std::getenv("AAM_FULL_PROMPT");
    if (prompt_text == nullptr) prompt_text = "The object is flying";
    auto prompt = loaded->tokenize(prompt_text);
    auto empty = loaded->tokenize("");
    if (!prompt || !empty) return 1;
    std::vector<std::int32_t> tokens;
    tokens.reserve(prompt->size() + empty->size());
    tokens.insert(tokens.end(), prompt->begin(), prompt->end());
    tokens.insert(tokens.end(), empty->begin(), empty->end());
    auto clip = implementation->clip_encode(tokens, 2U);
    if (!clip) return 1;
    auto latent = implementation->rf_sample(
        f0, indices.size(), *clip, noise, 64U, 3.0F);
    if (!latent) return 1;
    okay = compare("full.rf_normalized", *latent, expected_rf) && okay;
    for (std::size_t token = 0; token < indices.size(); ++token) {
        for (std::size_t channel = 0; channel < 64U; ++channel) {
            auto & value = (*latent)[token * 64U + channel];
            value = channel < 32U ? std::fma(value, f0_std, f0_mean)
                                  : std::fma(value, ft_std, ft_mean);
        }
    }
    okay = compare("full.latent", *latent, expected_latent) && okay;
    auto offsets = implementation->vae_decode_offsets(*embeddings, *latent, indices.size());
    if (!offsets) return 1;
    std::vector<float> output(expected.size());
    for (std::size_t frame = 0; frame < aam::default_frames; ++frame) {
        for (std::size_t point = 0; point < prepared->value.vertices.size(); ++point) {
            const std::size_t output_index = (frame * prepared->value.vertices.size() + point) * 3U;
            const std::size_t offset_index = point * 48U + frame * 3U;
            output[output_index] = prepared->value.vertices[point].x + (*offsets)[offset_index];
            output[output_index + 1U] = prepared->value.vertices[point].y + (*offsets)[offset_index + 1U];
            output[output_index + 2U] = prepared->value.vertices[point].z + (*offsets)[offset_index + 2U];
        }
    }
    okay = compare("full.output", output, expected) && okay;
    compare_motion_frames(
        output, expected, prepared->value.vertices.size(), aam::default_frames);
    if (const char * prefix = std::getenv("AAM_FULL_OUTPUT_PREFIX")) {
        okay = save_comparison_glbs(
            prefix, prepared->value, output, expected,
            prepared->value.vertices.size(), aam::default_frames) && okay;
    }
    return okay ? 0 : 1;
}
