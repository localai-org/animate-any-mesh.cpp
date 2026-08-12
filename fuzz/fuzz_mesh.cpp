#include "aam/aam.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    static_assert(sizeof(aam::vec3) == 3U * sizeof(std::uint32_t));
    if (size < 2U || size > (1U << 20U)) return 0;

    const std::size_t vertex_count = 1U + static_cast<std::size_t>(data[0]);
    const std::size_t face_count = static_cast<std::size_t>(data[1]) * 2U;
    aam::mesh candidate;
    candidate.vertices.resize(vertex_count);
    candidate.faces.resize(face_count);

    std::size_t cursor = 2U;
    auto take_byte = [&]() {
        const std::uint8_t value = data[cursor % size];
        ++cursor;
        return value;
    };
    for (auto & vertex : candidate.vertices) {
        std::uint32_t bits[3]{};
        for (auto & value : bits) {
            value = static_cast<std::uint32_t>(take_byte()) |
                    static_cast<std::uint32_t>(take_byte()) << 8U |
                    static_cast<std::uint32_t>(take_byte()) << 16U |
                    static_cast<std::uint32_t>(take_byte()) << 24U;
        }
        std::memcpy(&vertex, bits, sizeof(vertex));
    }
    for (auto & face : candidate.faces) {
        for (auto & index : face) {
            const std::uint32_t raw = static_cast<std::uint32_t>(take_byte()) |
                                      static_cast<std::uint32_t>(take_byte()) << 8U;
            index = (take_byte() & 1U) == 0U
                ? raw % static_cast<std::uint32_t>(vertex_count)
                : raw;
        }
    }
    (void)aam::prepare_mesh(candidate, 256U);
    (void)aam::simplify_mesh(candidate, 3U + static_cast<std::size_t>(data[0] % 64U));
    return 0;
}
