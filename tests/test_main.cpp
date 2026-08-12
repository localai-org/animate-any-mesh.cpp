#include "aam/aam.hpp"
#include "pytorch_rng.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void append_u32(std::vector<std::byte> & output, std::uint32_t value) {
    if constexpr (std::endian::native == std::endian::big) value = std::byteswap(value);
    const auto * begin = reinterpret_cast<const std::byte *>(&value);
    output.insert(output.end(), begin, begin + sizeof(value));
}

void append_bytes(std::vector<std::byte> & output, const void * data, std::size_t bytes) {
    const auto * begin = reinterpret_cast<const std::byte *>(data);
    output.insert(output.end(), begin, begin + bytes);
}

std::vector<std::byte> triangle_glb() {
    std::string json = R"({"asset":{"version":"2.0"},"buffers":[{"byteLength":42}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"nodes":[{"mesh":0,"translation":[3,4,5]}],"scenes":[{"nodes":[0]}],"scene":0})";
    while ((json.size() % 4U) != 0U) json.push_back(' ');
    const std::array<float, 9> positions{0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    std::vector<std::byte> binary;
    append_bytes(binary, positions.data(), sizeof(positions));
    append_bytes(binary, indices.data(), sizeof(indices));
    while ((binary.size() % 4U) != 0U) binary.push_back(std::byte{0});

    std::vector<std::byte> output;
    append_u32(output, 0x46546C67U);
    append_u32(output, 2U);
    append_u32(output, static_cast<std::uint32_t>(12U + 8U + json.size() + 8U + binary.size()));
    append_u32(output, static_cast<std::uint32_t>(json.size()));
    append_u32(output, 0x4E4F534AU);
    append_bytes(output, json.data(), json.size());
    append_u32(output, static_cast<std::uint32_t>(binary.size()));
    append_u32(output, 0x004E4942U);
    append_bytes(output, binary.data(), binary.size());
    return output;
}

void test_glb() {
    const auto bytes = triangle_glb();
    auto parsed = aam::load_glb(bytes);
    check(parsed.has_value(), "valid triangle GLB parses");
    if (!parsed) return;
    check(parsed->vertices.size() == 3U, "GLB has three positions");
    check(parsed->faces.size() == 1U, "GLB has one triangle");
    check(parsed->vertices[0] == aam::vec3{3.0F, 4.0F, 5.0F},
          "GLB scene-node transform is applied");
    auto truncated = aam::load_glb(std::span{bytes}.first(bytes.size() - 1U));
    check(!truncated, "truncated GLB is rejected");
}

void test_prepare() {
    aam::mesh input;
    input.vertices = {
        {1.0F, 1.0F, 0.0F},
        {-1.0F, -1.0F, 0.0F},
        {1.0F, -1.0F, 0.0F},
        {1.0F, 1.0F, 0.0F},
    };
    input.faces = {{1, 2, 0}, {1, 2, 3}, {0, 3, 2}};
    auto prepared = aam::prepare_mesh(input);
    check(prepared.has_value(), "mesh preprocessing succeeds");
    if (!prepared) return;
    check(prepared->value.vertices.size() == 3U, "duplicate vertices merge");
    check(prepared->value.faces.size() == 1U, "duplicate and degenerate faces are removed");
    check(prepared->original_to_merged[0] == prepared->original_to_merged[3],
          "inverse vertex map preserves duplicate association");
    check(prepared->adjacency_offsets == std::vector<std::uint32_t>({0, 2, 4, 6}),
          "triangle CSR offsets are exact");
    check(std::ranges::all_of(prepared->value.vertices, [](const aam::vec3 & value) {
        return std::abs(value.x) <= 1.0F && std::abs(value.y) <= 1.0F && std::abs(value.z) <= 1.0F;
    }), "normalized coordinates are bounded");
}

void test_simplify() {
    constexpr std::uint32_t side = 33U;
    aam::mesh grid;
    grid.vertices.reserve(static_cast<std::size_t>(side) * side);
    for (std::uint32_t y = 0U; y < side; ++y) {
        for (std::uint32_t x = 0U; x < side; ++x) {
            const float z = std::sin(static_cast<float>(x) * 0.25F) *
                            std::cos(static_cast<float>(y) * 0.25F) * 0.1F;
            grid.vertices.push_back({static_cast<float>(x), static_cast<float>(y), z});
        }
    }
    for (std::uint32_t y = 0U; y + 1U < side; ++y) {
        for (std::uint32_t x = 0U; x + 1U < side; ++x) {
            const auto a = y * side + x;
            const auto b = a + 1U;
            const auto c = a + side;
            const auto d = c + 1U;
            grid.faces.push_back({a, b, d});
            grid.faces.push_back({a, d, c});
        }
    }
    constexpr std::size_t budget = 128U;
    auto first = aam::simplify_mesh(grid, budget);
    auto second = aam::simplify_mesh(grid, budget);
    check(first.has_value(), "dense mesh simplification succeeds");
    check(second.has_value(), "dense mesh simplification is repeatable");
    if (!first || !second) return;
    check(first->vertices.size() <= budget, "simplification strictly respects vertex budget");
    check(!first->faces.empty(), "simplification retains triangle faces");
    check(first->vertices == second->vertices && first->faces == second->faces,
          "simplification is deterministic");
    check(std::ranges::all_of(first->faces, [&](const aam::triangle & face) {
        return face[0] < first->vertices.size() && face[1] < first->vertices.size() &&
               face[2] < first->vertices.size();
    }), "simplified triangle indices remain in bounds");

    auto unchanged = aam::simplify_mesh(
        aam::mesh{{{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
                  {{0U, 1U, 2U}}}, 3U);
    check(unchanged.has_value() && unchanged->vertices.size() == 3U,
          "mesh already at budget is preserved");
    auto impossible = aam::simplify_mesh(grid, 2U);
    check(!impossible && impossible.error().code == aam::error_code::invalid_argument,
          "simplification rejects budgets too small for a triangle");
}

void test_fps() {
    const std::array<float, 5> points{0.0F, 1.0F, 2.0F, 8.0F, 9.0F};
    auto sampled = aam::farthest_point_sample(points, 5U, 1U, 3U);
    check(sampled.has_value(), "FPS succeeds");
    if (sampled) {
        check(*sampled == std::vector<std::uint32_t>({0, 4, 2}),
              "FPS starts at zero and uses first-index tie breaking");
    }
    auto bad = aam::farthest_point_sample(points, 5U, 1U, 6U);
    check(!bad, "FPS rejects sample count greater than point count");

    constexpr std::size_t point_count = 4097U;
    constexpr std::size_t dimensions = 256U;
    constexpr std::size_t sample_count = 17U;
    std::vector<float> large_points(point_count * dimensions);
    std::uint32_t state = 0x12345678U;
    for (auto & value : large_points) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(state >> 8U) / 16777216.0F;
    }
    std::vector<std::uint32_t> expected;
    expected.reserve(sample_count);
    expected.push_back(0U);
    std::vector<float> closest(point_count, std::numeric_limits<float>::infinity());
    for (std::size_t iteration = 1U; iteration < sample_count; ++iteration) {
        const auto latest = static_cast<std::size_t>(expected.back());
        std::size_t farthest = 0U;
        float farthest_distance = -1.0F;
        for (std::size_t point = 0; point < point_count; ++point) {
            float distance = 0.0F;
            for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                const float delta = large_points[point * dimensions + dimension] -
                                    large_points[latest * dimensions + dimension];
                distance = std::fma(delta, delta, distance);
            }
            closest[point] = std::min(closest[point], distance);
            if (closest[point] > farthest_distance) {
                farthest = point;
                farthest_distance = closest[point];
            }
        }
        expected.push_back(static_cast<std::uint32_t>(farthest));
    }
    auto parallel = aam::farthest_point_sample(
        large_points, point_count, dimensions, sample_count);
    check(parallel.has_value() && *parallel == expected,
          "parallel FPS exactly matches the serial ordered reduction");
}

void test_pytorch_cuda_rng() {
    const aam::detail::pytorch_cuda_generator generator{666U};
    const std::array<float, 16> expected{
        -0.697696447F, -2.893378258F, -2.077694178F, -0.243898198F,
         0.329254061F, -0.710392654F, -1.335972667F, -0.896324992F,
         0.301725298F,  0.868598759F,  0.625175118F, -0.307230920F,
         0.996807456F,  0.709128797F, -0.272168934F, -0.537082970F,
    };
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        check(std::abs(generator.normal(index, 1020U * 64U) - expected[index]) < 2.0e-3F,
              "PyTorch CUDA Philox stream matches captured seed 666");
    }
    check(generator.replacement_index(0U, 8160U) == 1617U,
          "PyTorch CUDA replacement index matches captured seed 666");
}

void test_animation_export() {
    aam::mesh topology;
    topology.vertices = {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    topology.faces = {{0U, 1U, 2U}};
    aam::animation animation;
    animation.frames = 2U;
    animation.vertices_per_frame = 3U;
    animation.positions = topology.vertices;
    animation.positions.insert(animation.positions.end(), {
        {0.0F, 0.0F, 0.25F}, {1.0F, 0.0F, 0.25F}, {0.0F, 1.0F, 0.25F}});
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output = std::filesystem::temp_directory_path() /
        ("aam-animation-export-test-" + std::to_string(nonce) + ".glb");
    std::error_code ignored;
    std::filesystem::remove(output, ignored);
    auto saved = aam::save_animation_glb_file(output, topology, animation, 24.0F);
    check(saved.has_value(), "animated GLB export succeeds");
    if (saved) {
        auto loaded = aam::load_glb_file(output);
        check(loaded.has_value(), "exported animated GLB is structurally readable");
        if (loaded) {
            check(loaded->vertices.size() == topology.vertices.size(),
                  "exported GLB preserves vertex count");
            check(loaded->faces == topology.faces, "exported GLB preserves topology");
        }
    }
    std::filesystem::remove(output, ignored);
}

} // namespace

int main() {
    test_glb();
    test_prepare();
    test_simplify();
    test_fps();
    test_pytorch_cuda_rng();
    test_animation_export();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
