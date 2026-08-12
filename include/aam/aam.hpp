#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(AAM_SHARED)
#  if defined(_WIN32) && !defined(__MINGW32__)
#    if defined(AAM_BUILD)
#      define AAM_API __declspec(dllexport)
#    else
#      define AAM_API __declspec(dllimport)
#    endif
#  else
#    define AAM_API __attribute__((visibility("default")))
#  endif
#else
#  define AAM_API
#endif

namespace aam {

inline constexpr std::string_view version = "0.1.0";
inline constexpr std::size_t default_max_vertices = 4096;
inline constexpr std::size_t default_frames = 16;

enum class error_code : std::uint8_t {
    invalid_argument,
    invalid_format,
    limit_exceeded,
    io,
    incompatible_model,
    backend_unavailable,
    allocation,
    compute,
};

struct error {
    error_code code = error_code::invalid_argument;
    std::string message;
};

template<class T>
using result = std::expected<T, error>;

struct vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;

    [[nodiscard]] constexpr bool operator==(const vec3 &) const noexcept = default;
};

using triangle = std::array<std::uint32_t, 3>;

struct mesh {
    std::vector<vec3> vertices;
    std::vector<triangle> faces;
};

struct normalization {
    vec3 center;
    float scale = 1.0F;
};

struct prepared_mesh {
    mesh value;
    normalization transform;
    std::vector<std::uint32_t> original_to_merged;
    std::vector<std::uint32_t> adjacency_offsets;
    std::vector<std::uint32_t> adjacency_indices;
};

struct mesh_limits {
    std::size_t max_file_bytes = 256U * 1024U * 1024U;
    std::size_t max_json_bytes = 16U * 1024U * 1024U;
    std::size_t max_vertices = 1U << 20U;
    std::size_t max_faces = 1U << 21U;
};

enum class device_kind : std::uint8_t { automatic, cpu, vulkan };

struct runtime_options {
    device_kind device = device_kind::automatic;
    std::uint32_t threads = 0;
    std::filesystem::path backend_directory;
    // Vulkan reductions can perturb farthest-point tie ordering enough to pair
    // seeded RF noise with different trajectories. Keep the small FPS
    // embedding/order computation on the dynamically loaded CPU backend when
    // the model carries its parity-sensitive VAE weights in F32.
    bool strict_fps_parity = true;
};

struct generation_options {
    std::uint64_t seed = 666;
    std::size_t trajectories = 0;
    std::size_t steps = 64;
    float guidance_scale = 3.0F;
    std::size_t max_vertices = default_max_vertices;
};

struct animation {
    std::size_t frames = 0;
    std::size_t vertices_per_frame = 0;
    std::vector<vec3> positions;
};

struct text_embedding {
    std::size_t tokens = 0;
    std::size_t width = 0;
    std::vector<float> values; // row-major [tokens, width]
};

class AAM_API model final {
public:
    model(model &&) noexcept;
    model & operator=(model &&) noexcept;
    model(const model &) = delete;
    model & operator=(const model &) = delete;
    ~model();

    [[nodiscard]] static result<model> load(
        const std::filesystem::path & gguf_path,
        const runtime_options & options = {});

    [[nodiscard]] result<animation> generate(
        const mesh & source,
        std::string_view prompt,
        const generation_options & options = {}) const;
    // Reuse validated preprocessing when generating multiple animations from
    // the same topology. options.max_vertices is already reflected in source.
    [[nodiscard]] result<animation> generate(
        const prepared_mesh & source,
        std::string_view prompt,
        const generation_options & options = {}) const;

    // Exposed for reproducible layer validation and applications that cache
    // prompt conditioning. Production animation calls this internally.
    [[nodiscard]] result<std::vector<std::int32_t>> tokenize(
        std::string_view prompt) const;
    [[nodiscard]] result<text_embedding> encode_text(
        std::string_view prompt) const;
    [[nodiscard]] result<std::vector<text_embedding>> encode_text_blocks(
        std::string_view prompt) const;

    [[nodiscard]] std::string_view backend_name() const noexcept;

private:
    friend struct validation_access;
    struct impl;
    explicit model(std::unique_ptr<impl>) noexcept;
    std::unique_ptr<impl> impl_;
};

[[nodiscard]] AAM_API result<mesh> load_glb(
    std::span<const std::byte> bytes,
    const mesh_limits & limits = {});

[[nodiscard]] AAM_API result<mesh> load_glb_file(
    const std::filesystem::path & path,
    const mesh_limits & limits = {});

[[nodiscard]] AAM_API result<prepared_mesh> prepare_mesh(
    const mesh & source,
    std::size_t max_vertices = default_max_vertices);

// Return a topology with no more than max_vertices referenced vertices. Meshes
// already within budget are preserved; larger meshes are deterministically
// deduplicated and simplified while retaining valid triangle winding.
[[nodiscard]] AAM_API result<mesh> simplify_mesh(
    const mesh & source,
    std::size_t max_vertices = default_max_vertices);

[[nodiscard]] AAM_API result<void> save_animation_glb_file(
    const std::filesystem::path & path,
    const mesh & topology,
    const animation & value,
    float frames_per_second = 24.0F);

[[nodiscard]] AAM_API result<std::vector<std::uint32_t>> farthest_point_sample(
    std::span<const float> points,
    std::size_t point_count,
    std::size_t dimensions,
    std::size_t sample_count);

[[nodiscard]] AAM_API std::vector<std::string> available_backends(
    const std::filesystem::path & backend_directory = {});

} // namespace aam
