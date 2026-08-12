#include "aam/aam.hpp"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct gguf_deleter { void operator()(gguf_context * value) const { gguf_free(value); } };
struct ggml_deleter { void operator()(ggml_context * value) const { ggml_free(value); } };

bool compare(std::span<const float> got, std::span<const float> expected,
             float atol, float rtol) {
    if (got.size() != expected.size() || got.empty()) return false;
    float max_difference = 0.0F;
    std::size_t worst = 0;
    std::size_t failures = 0;
    double squared_difference = 0.0;
    double squared_reference = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float difference = std::abs(got[i] - expected[i]);
        if (difference > max_difference) { max_difference = difference; worst = i; }
        if (difference > atol + rtol * std::abs(expected[i])) ++failures;
        squared_difference += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected[i]) * expected[i];
    }
    std::cout << "clip.final: max_abs=" << max_difference
              << " rel_l2=" << std::sqrt(squared_difference / squared_reference)
              << " worst=" << worst << " got=" << got[worst]
              << " ref=" << expected[worst] << " failures=" << failures << '\n';
    return failures == 0U;
}

} // namespace

int main() {
    const char * model_path = std::getenv("AAM_TEST_GGUF");
    const char * reference_path = std::getenv("AAM_CLIP_REFERENCE");
    if (model_path == nullptr || reference_path == nullptr) {
        std::cout << "SKIP: set AAM_TEST_GGUF and AAM_CLIP_REFERENCE\n";
        return 77;
    }

    ggml_context * raw_context = nullptr;
    gguf_init_params parameters{};
    parameters.no_alloc = false;
    parameters.ctx = &raw_context;
    std::unique_ptr<gguf_context, gguf_deleter> reference(
        gguf_init_from_file(reference_path, parameters));
    std::unique_ptr<ggml_context, ggml_deleter> context(raw_context);
    if (!reference || !context) {
        std::cerr << "failed to load CLIP reference\n";
        return 1;
    }
    auto * token_tensor = ggml_get_tensor(context.get(), "tokens");
    auto * output_tensor = ggml_get_tensor(context.get(), "clip.final");
    if (token_tensor == nullptr || output_tensor == nullptr ||
        token_tensor->type != GGML_TYPE_I32 || output_tensor->type != GGML_TYPE_F32) {
        std::cerr << "reference has invalid tensors\n";
        return 1;
    }
    std::vector<std::int32_t> expected_tokens(static_cast<std::size_t>(ggml_nelements(token_tensor)));
    std::memcpy(expected_tokens.data(), token_tensor->data, ggml_nbytes(token_tensor));
    std::vector<float> expected_output(static_cast<std::size_t>(ggml_nelements(output_tensor)));
    std::memcpy(expected_output.data(), output_tensor->data, ggml_nbytes(output_tensor));

    aam::runtime_options runtime;
    runtime.device = aam::device_kind::cpu;
    auto model = aam::model::load(model_path, runtime);
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    auto actual_tokens = model->tokenize("The object is flying");
    if (!actual_tokens || *actual_tokens != expected_tokens) {
        std::cerr << "CLIP token IDs differ from transformers.CLIPTokenizer\n";
        return 1;
    }
    const std::array unicode_cases{
        std::pair<std::string_view, std::vector<std::int32_t>>{
            "Café déjà-vu — １２٣ cats",
            {49406, 15304, 25466, 73, 21259, 268, 13230, 2005, 171, 120, 495,
             171, 120, 496, 149, 352, 3989, 49407}},
        std::pair<std::string_view, std::vector<std::int32_t>>{
            "你好，世界! 🚀 Ⅻ",
            {49406, 47466, 510, 29290, 377, 171, 120, 490, 19759, 500, 163,
             243, 490, 256, 13542, 158, 227, 375, 49407}},
        std::pair<std::string_view, std::vector<std::int32_t>>{
            "İSTANBUL &amp; Straße",
            {49406, 328, 16384, 11231, 261, 6259, 282, 1894, 127, 253, 324, 49407}},
    };
    for (const auto & [prompt, expected_prefix] : unicode_cases) {
        auto unicode_tokens = model->tokenize(prompt);
        if (!unicode_tokens || unicode_tokens->size() < expected_prefix.size() ||
            !std::equal(expected_prefix.begin(), expected_prefix.end(), unicode_tokens->begin())) {
            std::cerr << "CLIP multilingual token IDs differ for: " << prompt << '\n';
            if (unicode_tokens) {
                for (std::size_t index = 0; index < expected_prefix.size(); ++index) {
                    std::cerr << (*unicode_tokens)[index] << (index + 1U == expected_prefix.size() ? '\n' : ' ');
                }
            }
            return 1;
        }
    }
    auto encoded = model->encode_text("The object is flying");
    if (!encoded) {
        std::cerr << encoded.error().message << '\n';
        return 1;
    }
    bool okay = compare(encoded->values, expected_output, 3.5e-3F, 1.0e-4F);
    auto blocks = model->encode_text_blocks("The object is flying");
    if (!blocks || blocks->size() != 13U) {
        std::cerr << "failed to obtain 12 CLIP block outputs and final norm\n";
        return 1;
    }
    for (std::size_t i = 0; i < 12U; ++i) {
        const std::string name = "clip.layer." + std::to_string(i);
        auto * tensor = ggml_get_tensor(context.get(), name.c_str());
        if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
            std::cerr << "reference lacks " << name << '\n';
            return 1;
        }
        std::span<const float> expected(static_cast<const float *>(tensor->data),
                                        static_cast<std::size_t>(ggml_nelements(tensor)));
        std::cout << "block " << i << ' ';
        okay = compare((*blocks)[i].values, expected, 3.5e-2F, 1.0e-4F) && okay;
    }
    return okay ? 0 : 1;
}
