#pragma once

#include "internal.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aam::detail {

class clip_tokenizer final {
public:
    [[nodiscard]] result<void> load(const gguf_context * context);
    [[nodiscard]] result<std::vector<std::int32_t>> encode(
        std::string_view text, std::size_t context_length = 77U) const;

private:
    struct pair_hash {
        [[nodiscard]] std::size_t operator()(
            const std::pair<std::string, std::string> & value) const noexcept;
    };

    std::unordered_map<std::string, std::int32_t> vocabulary_;
    std::unordered_map<std::pair<std::string, std::string>, std::int32_t, pair_hash> merges_;
    std::int32_t bos_ = 49406;
    std::int32_t eos_ = 49407;
};

} // namespace aam::detail
