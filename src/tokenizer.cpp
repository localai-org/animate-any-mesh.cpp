#include "tokenizer.hpp"

#include <utf8proc.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <list>
#include <queue>

namespace aam::detail {
namespace {

[[nodiscard]] std::uint32_t decode_one(std::string_view text, std::size_t & offset) {
    const auto lead = static_cast<std::uint8_t>(text[offset]);
    if (lead < 0x80U) {
        ++offset;
        return lead;
    }
    std::size_t length = 0;
    std::uint32_t value = 0;
    if ((lead & 0xE0U) == 0xC0U) { length = 2; value = lead & 0x1FU; }
    else if ((lead & 0xF0U) == 0xE0U) { length = 3; value = lead & 0x0FU; }
    else if ((lead & 0xF8U) == 0xF0U) { length = 4; value = lead & 0x07U; }
    else { ++offset; return 0x110000U; }
    if (offset + length > text.size()) { ++offset; return 0x110000U; }
    for (std::size_t i = 1; i < length; ++i) {
        const auto byte = static_cast<std::uint8_t>(text[offset + i]);
        if ((byte & 0xC0U) != 0x80U) { ++offset; return 0x110000U; }
        value = (value << 6U) | (byte & 0x3FU);
    }
    const bool overlong = (length == 2U && value < 0x80U) ||
                          (length == 3U && value < 0x800U) ||
                          (length == 4U && value < 0x10000U);
    if (overlong || value > 0x10FFFFU || (value >= 0xD800U && value <= 0xDFFFU)) {
        ++offset;
        return 0x110000U;
    }
    offset += length;
    return value;
}

[[nodiscard]] bool unicode_space(std::uint32_t cp) {
    return cp == 0x20U || (cp >= 0x09U && cp <= 0x0DU) || cp == 0x85U || cp == 0xA0U ||
           cp == 0x1680U || (cp >= 0x2000U && cp <= 0x200AU) || cp == 0x2028U ||
           cp == 0x2029U || cp == 0x202FU || cp == 0x205FU || cp == 0x3000U;
}

enum class character_class : std::uint8_t { letter, number, punctuation, space, invalid };

[[nodiscard]] character_class classify(std::uint32_t cp) {
    if (cp > 0x10FFFFU) return character_class::invalid;
    if (unicode_space(cp)) return character_class::space;
    const auto category = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
    switch (category) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO: return character_class::letter;
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO: return character_class::number;
    default: return character_class::punctuation;
    }
}

[[nodiscard]] std::string utf8(std::uint32_t cp);

[[nodiscard]] bool cjk_ideograph(std::uint32_t cp) {
    return (cp >= 0x4E00U && cp <= 0x9FFFU) || (cp >= 0x3400U && cp <= 0x4DBFU) ||
           (cp >= 0x20000U && cp <= 0x2A6DFU) || (cp >= 0x2A700U && cp <= 0x2B73FU) ||
           (cp >= 0x2B740U && cp <= 0x2B81FU) || (cp >= 0x2B820U && cp <= 0x2CEAFU) ||
           (cp >= 0xF900U && cp <= 0xFAFFU) || (cp >= 0x2F800U && cp <= 0x2FA1FU);
}

[[nodiscard]] std::string clean_text(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    bool pending_space = false;
    std::size_t offset = 0;
    while (offset < input.size()) {
        const std::size_t begin = offset;
        const auto cp = decode_one(input, offset);
        if (classify(cp) == character_class::invalid) {
            continue;
        }
        if (unicode_space(cp)) {
            pending_space = !output.empty();
            continue;
        }
        // transformers 4.55 falls back to its BasicTokenizer when ftfy is not
        // installed (as in the pinned upstream environment), which surrounds
        // each CJK ideograph with whitespace before CLIP's regex pass.
        if (cjk_ideograph(cp)) {
            if (!output.empty() && output.back() != ' ') output.push_back(' ');
            output.append(input.substr(begin, offset - begin));
            pending_space = true;
            continue;
        }
        if (pending_space) {
            output.push_back(' ');
            pending_space = false;
        }
        if (cp == 0x0130U) {
            output.append("i\xCC\x87"); // Python lower(): LATIN I WITH DOT -> i + combining dot
        } else {
            const auto lowered = utf8proc_tolower(static_cast<utf8proc_int32_t>(cp));
            if (lowered < 0) output.append(input.substr(begin, offset - begin));
            else output.append(utf8(static_cast<std::uint32_t>(lowered)));
        }
    }
    return output;
}

[[nodiscard]] std::vector<std::string_view> pretokenize(std::string_view text) {
    std::vector<std::string_view> output;
    std::size_t offset = 0;
    while (offset < text.size()) {
        if (text[offset] == ' ') { ++offset; continue; }
        const std::size_t begin = offset;
        std::size_t probe = offset;
        const auto cp = decode_one(text, probe);
        const auto cls = classify(cp);
        offset = probe;

        if (text[begin] == '\'' && offset < text.size()) {
            constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            bool found = false;
            for (const auto suffix : suffixes) {
                if (text.substr(offset).starts_with(suffix)) {
                    output.emplace_back(text.data() + begin, 1U + suffix.size());
                    offset += suffix.size();
                    found = true;
                    break;
                }
            }
            if (found) continue;
        }

        if (cls == character_class::number) {
            output.emplace_back(text.data() + begin, offset - begin);
            continue; // CLIP's pattern emits one numeric codepoint at a time
        }
        while (offset < text.size()) {
            if (text[offset] == ' ') break;
            std::size_t next = offset;
            const auto next_cp = decode_one(text, next);
            if (classify(next_cp) != cls) break;
            offset = next;
        }
        output.emplace_back(text.data() + begin, offset - begin);
    }
    return output;
}

[[nodiscard]] std::string utf8(std::uint32_t cp) {
    std::string out;
    if (cp <= 0x7FU) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
    return out;
}

[[nodiscard]] std::array<std::string, 256> byte_encoder() {
    std::array<std::string, 256> result;
    std::uint32_t shifted = 0;
    for (std::uint32_t byte = 0; byte < 256U; ++byte) {
        const bool direct = (byte >= '!' && byte <= '~') || (byte >= 0xA1U && byte <= 0xACU) ||
                            (byte >= 0xAEU && byte <= 0xFFU);
        result[byte] = utf8(direct ? byte : 0x100U + shifted++);
    }
    return result;
}

[[nodiscard]] std::string decode_byte_unicode(std::string_view input) {
    static const auto encoded = byte_encoder();
    static const auto reverse = [] {
        std::unordered_map<std::string, char> table;
        const auto values = byte_encoder();
        for (std::size_t i = 0; i < values.size(); ++i) {
            table.emplace(values[i], static_cast<char>(i));
        }
        return table;
    }();
    std::string output;
    for (std::size_t offset = 0; offset < input.size();) {
        const std::size_t begin = offset;
        (void)decode_one(input, offset);
        const auto it = reverse.find(std::string{input.substr(begin, offset - begin)});
        if (it == reverse.end()) return {};
        output.push_back(it->second);
    }
    (void)encoded;
    return output;
}

} // namespace

std::size_t clip_tokenizer::pair_hash::operator()(
    const std::pair<std::string, std::string> & value) const noexcept {
    const auto first = std::hash<std::string>{}(value.first);
    const auto second = std::hash<std::string>{}(value.second);
    return first ^ (second + 0x9E3779B97F4A7C15ULL + (first << 6U) + (first >> 2U));
}

result<void> clip_tokenizer::load(const gguf_context * context) {
    if (context == nullptr) {
        return std::unexpected(fail(error_code::invalid_argument, "null GGUF tokenizer context"));
    }
    const auto token_key = gguf_find_key(context, "tokenizer.ggml.tokens");
    const auto merge_key = gguf_find_key(context, "tokenizer.ggml.merges");
    if (token_key < 0 || merge_key < 0) {
        return std::unexpected(fail(error_code::incompatible_model,
            "GGUF has no CLIP tokens or merges"));
    }
    const auto token_count = gguf_get_arr_n(context, token_key);
    if (token_count == 0U || token_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return std::unexpected(fail(error_code::incompatible_model, "invalid CLIP vocabulary size"));
    }
    vocabulary_.clear();
    vocabulary_.reserve(token_count);
    for (std::size_t i = 0; i < token_count; ++i) {
        const char * encoded = gguf_get_arr_str(context, token_key, i);
        if (encoded == nullptr) {
            return std::unexpected(fail(error_code::incompatible_model, "null CLIP vocabulary entry"));
        }
        auto raw = decode_byte_unicode(encoded);
        if (raw.empty() && encoded[0] != '\0') raw = encoded; // special tokens
        vocabulary_.emplace(std::move(raw), static_cast<std::int32_t>(i));
    }
    const auto merge_count = gguf_get_arr_n(context, merge_key);
    merges_.clear();
    merges_.reserve(merge_count);
    for (std::size_t i = 0; i < merge_count; ++i) {
        const std::string encoded = gguf_get_arr_str(context, merge_key, i);
        const auto separator = encoded.find(' ');
        if (separator == std::string::npos) continue;
        auto left = decode_byte_unicode(std::string_view{encoded}.substr(0, separator));
        auto right = decode_byte_unicode(std::string_view{encoded}.substr(separator + 1U));
        merges_.emplace(std::make_pair(std::move(left), std::move(right)), static_cast<std::int32_t>(i));
    }
    const auto bos_key = gguf_find_key(context, "tokenizer.ggml.bos_token_id");
    const auto eos_key = gguf_find_key(context, "tokenizer.ggml.eos_token_id");
    if (bos_key >= 0) bos_ = static_cast<std::int32_t>(gguf_get_val_u32(context, bos_key));
    if (eos_key >= 0) eos_ = static_cast<std::int32_t>(gguf_get_val_u32(context, eos_key));
    return {};
}

result<std::vector<std::int32_t>> clip_tokenizer::encode(
    std::string_view text, std::size_t context_length) const {
    if (context_length < 2U || context_length > 4096U || vocabulary_.empty()) {
        return std::unexpected(fail(error_code::invalid_argument,
            "invalid CLIP context length or unloaded tokenizer"));
    }
    const auto cleaned = clean_text(text);
    std::vector<std::int32_t> result_tokens;
    result_tokens.reserve(context_length);
    result_tokens.push_back(bos_);

    for (const auto word : pretokenize(cleaned)) {
        std::list<std::string> symbols;
        for (const auto byte : word) {
            symbols.emplace_back(1U, byte);
        }
        if (!symbols.empty()) {
            symbols.back().append("</w>");
        }
        while (symbols.size() > 1U) {
            auto best = merges_.end();
            auto best_left = symbols.end();
            for (auto left = symbols.begin(), right = std::next(left); right != symbols.end(); ++left, ++right) {
                const auto candidate = merges_.find({*left, *right});
                if (candidate != merges_.end() && (best == merges_.end() || candidate->second < best->second)) {
                    best = candidate;
                    best_left = left;
                }
            }
            if (best == merges_.end()) break;
            auto right = std::next(best_left);
            best_left->append(*right);
            symbols.erase(right);
        }
        for (const auto & symbol : symbols) {
            const auto token = vocabulary_.find(symbol);
            if (token == vocabulary_.end()) {
                return std::unexpected(fail(error_code::invalid_format,
                    "CLIP BPE produced a symbol absent from the vocabulary"));
            }
            if (result_tokens.size() + 1U >= context_length) break;
            result_tokens.push_back(token->second);
        }
        if (result_tokens.size() + 1U >= context_length) break;
    }
    result_tokens.push_back(eos_);
    result_tokens.resize(context_length, eos_);
    return result_tokens;
}

} // namespace aam::detail
