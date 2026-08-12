#include "aam/aam.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    if (size < 2U || size > (1U << 20U)) return 0;
    const std::size_t dimensions = 1U + data[0] % 32U;
    const std::size_t floats = (size - 1U) / sizeof(float);
    const std::size_t points = floats / dimensions;
    if (points == 0U) return 0;
    std::vector<float> values(points * dimensions);
    std::memcpy(values.data(), data + 1U, values.size() * sizeof(float));
    const std::size_t samples = 1U + data[1] % std::min<std::size_t>(points, 64U);
    (void)aam::farthest_point_sample(values, points, dimensions, samples);
    return 0;
}
