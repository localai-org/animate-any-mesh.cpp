#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aam::detail {

// AnimateAnyMesh seeds CUDA before constructing its fixed RF architecture.
// PyTorch 2.7.1/CUDA 12.8 has advanced the CUDA Philox offset to 1616 after
// that construction. Sampling the replacement FPS indices advances it by four
// more words, so RF noise starts at 1620. Reproducing this stream makes a seed
// select the same diffusion sample without linking against CUDA or PyTorch.
//
// This is an independent implementation of the published Philox4x32-10
// counter generator plus a Box-Muller transform. It intentionally models the
// contiguous F32 CUDA kernel used by torch.randn for the workload sizes here.
class pytorch_cuda_generator final {
public:
    explicit constexpr pytorch_cuda_generator(std::uint64_t seed) noexcept
        : seed_(seed) {}

    [[nodiscard]] std::size_t replacement_index(
        std::size_t linear_index, std::size_t bound) const noexcept {
        if (bound == 0U) return 0U;
        const auto words = block(linear_index, model_construction_offset / 4U);
        constexpr float inverse = 1.0F / 4294967296.0F;
        float uniform = (static_cast<float>(words[0]) + 1.0F) * inverse;
        if (uniform == 1.0F) uniform = 0.0F;
        const auto value = static_cast<std::size_t>(
            uniform * static_cast<float>(bound));
        return value < bound ? value : bound - 1U;
    }

    [[nodiscard]] float normal(
        std::size_t linear_index, std::size_t total_elements) const noexcept {
        // PyTorch caps this CUDA grid at SMs * (max threads per SM / 256).
        // The pinned comparison GPU has 70 SMs and 1536 threads/SM.
        constexpr std::size_t block_threads = 256U;
        constexpr std::size_t reference_grid_limit = 70U * (1536U / block_threads);
        const std::size_t grid = std::min(
            (total_elements + block_threads - 1U) / block_threads,
            reference_grid_limit);
        const std::size_t threads = std::max<std::size_t>(1U, grid * block_threads);
        const std::size_t iteration_span = threads * 4U;
        const std::size_t iteration = linear_index / iteration_span;
        const std::size_t within_iteration = linear_index % iteration_span;
        const std::size_t lane = within_iteration / threads;
        const std::size_t thread = within_iteration % threads;
        const auto words = block(thread, noise_offset / 4U + iteration);
        const std::size_t pair = lane < 2U ? 0U : 2U;
        constexpr float inverse = 1.0F / 4294967296.0F;
        constexpr float half_inverse = 0.5F / 4294967296.0F;
        constexpr float inverse_two_pi =
            6.283185307179586476925286766559F / 4294967296.0F;
        const float u = std::fma(
            static_cast<float>(words[pair]), inverse, half_inverse);
        const float angle = std::fma(
            static_cast<float>(words[pair + 1U]), inverse_two_pi,
            inverse_two_pi * 0.5F);
        const float radius = std::sqrt(-2.0F * std::log(u));
        return radius * ((lane & 1U) == 0U ? std::sin(angle) : std::cos(angle));
    }

    static constexpr std::uint64_t model_construction_offset = 1616U;
    static constexpr std::uint64_t noise_offset = model_construction_offset + 4U;

private:
    [[nodiscard]] std::array<std::uint32_t, 4> block(
        std::uint64_t subsequence, std::uint64_t counter) const noexcept {
        std::array<std::uint32_t, 4> value{
            static_cast<std::uint32_t>(counter),
            static_cast<std::uint32_t>(counter >> 32U),
            static_cast<std::uint32_t>(subsequence),
            static_cast<std::uint32_t>(subsequence >> 32U),
        };
        std::uint32_t key0 = static_cast<std::uint32_t>(seed_);
        std::uint32_t key1 = static_cast<std::uint32_t>(seed_ >> 32U);
        for (unsigned round = 0U; round < 10U; ++round) {
            const std::uint64_t first =
                static_cast<std::uint64_t>(0xD2511F53U) * value[0];
            const std::uint64_t second =
                static_cast<std::uint64_t>(0xCD9E8D57U) * value[2];
            value = {
                static_cast<std::uint32_t>(second >> 32U) ^ value[1] ^ key0,
                static_cast<std::uint32_t>(second),
                static_cast<std::uint32_t>(first >> 32U) ^ value[3] ^ key1,
                static_cast<std::uint32_t>(first),
            };
            key0 += 0x9E3779B9U;
            key1 += 0xBB67AE85U;
        }
        return value;
    }

    std::uint64_t seed_;
};

} // namespace aam::detail
