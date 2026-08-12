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
    const char * reference_path = std::getenv("AAM_RF_REFERENCE");
    if (model_path == nullptr || reference_path == nullptr) {
        std::cout << "SKIP: set AAM_TEST_GGUF and AAM_RF_REFERENCE\n";
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

    const auto tokens = tensor_span<std::int32_t>(context.get(), "tokens", GGML_TYPE_I32);
    const auto input = tensor_span<float>(context.get(), "input.x", GGML_TYPE_F32);
    const auto times = tensor_span<float>(context.get(), "input.t", GGML_TYPE_F32);
    const auto reference_clip = tensor_span<float>(context.get(), "clip.final", GGML_TYPE_F32);
    if (tokens.size() != times.size() * 77U || input.size() % (times.size() * 64U) != 0U) return 1;
    const std::size_t token_count = input.size() / (times.size() * 64U);

    aam::runtime_options options;
    options.device = aam::device_kind::cpu;
    auto loaded = aam::model::load(model_path, options);
    if (!loaded) { std::cerr << loaded.error().message << '\n'; return 1; }
    auto * implementation = aam::validation_access::get(*loaded);
    auto actual_clip = implementation->clip_encode(tokens, times.size());
    if (!actual_clip) { std::cerr << actual_clip.error().message << '\n'; return 1; }
    bool okay = compare("rf.clip", *actual_clip, reference_clip, 3.5e-3F, 2.0e-4F);

    const std::vector<float> condition_mask{1.0F, 0.0F};
    aam::detail::rf_trace trace;
    auto output = implementation->rf_forward(
        input, token_count, reference_clip, times, condition_mask, &trace);
    if (!output) { std::cerr << output.error().message << '\n'; return 1; }
    okay = compare("rf.time.projected", trace.time_projected,
        tensor_span<float>(context.get(), "time.projected", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F) && okay;
    okay = compare("rf.text.projected", trace.text_projected,
        tensor_span<float>(context.get(), "text.projected", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F) && okay;
    okay = compare("rf.x.initial", trace.x_initial,
        tensor_span<float>(context.get(), "x.initial", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F) && okay;
    okay = compare("rf.text.initial", trace.text_initial,
        tensor_span<float>(context.get(), "text.initial", GGML_TYPE_F32), 2.0e-4F, 2.0e-4F) && okay;
    if (trace.blocks.size() != 48U) return 1;
    for (std::size_t block = 0; block < 12U; ++block) {
        const auto prefix = "block." + std::to_string(block);
        const std::array names{
            prefix + ".attention.x", prefix + ".attention.text",
            prefix + ".x", prefix + ".text"};
        for (std::size_t stage = 0; stage < names.size(); ++stage) {
            okay = compare(names[stage], trace.blocks[block * 4U + stage],
                tensor_span<float>(context.get(), names[stage].c_str(), GGML_TYPE_F32),
                1.5e-3F, 5.0e-4F) && okay;
        }
    }
    okay = compare("rf.final.norm", trace.final_norm,
        tensor_span<float>(context.get(), "final.norm", GGML_TYPE_F32), 1.5e-3F, 5.0e-4F) && okay;
    okay = compare("rf.output", *output,
        tensor_span<float>(context.get(), "output", GGML_TYPE_F32), 1.5e-3F, 5.0e-4F) && okay;
    const auto sample_f0 = tensor_span<float>(context.get(), "sample.f0", GGML_TYPE_F32);
    const auto sample_noise = tensor_span<float>(context.get(), "sample.noise", GGML_TYPE_F32);
    const auto sample_expected = tensor_span<float>(context.get(), "sample.output", GGML_TYPE_F32);
    if (!sample_f0.empty()) {
        const std::size_t sample_tokens = sample_f0.size() / 32U;
        auto sample = implementation->rf_sample(
            sample_f0, sample_tokens, reference_clip, sample_noise, 4U, 3.0F);
        if (!sample) { std::cerr << sample.error().message << '\n'; return 1; }
        okay = compare("rf.sample.midpoint", *sample, sample_expected,
            3.0e-3F, 8.0e-4F) && okay;
    }
    return okay ? 0 : 1;
}
