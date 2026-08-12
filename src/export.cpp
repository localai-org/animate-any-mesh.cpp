#include "internal.hpp"

#include <nlohmann/json.hpp>

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>

namespace aam {
namespace {

void append_u32(std::vector<std::byte> & output, std::uint32_t value) {
    if constexpr (std::endian::native == std::endian::big) value = std::byteswap(value);
    const auto * bytes = reinterpret_cast<const std::byte *>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(value));
}

void append_f32(std::vector<std::byte> & output, float value) {
    append_u32(output, std::bit_cast<std::uint32_t>(value));
}

void align4(std::vector<std::byte> & output) {
    while ((output.size() % 4U) != 0U) output.push_back(std::byte{0});
}

struct binary_view {
    std::size_t offset = 0;
    std::size_t length = 0;
};

} // namespace

result<void> save_animation_glb_file(
    const std::filesystem::path & path, const mesh & topology,
    const animation & value, float frames_per_second) {
    if (path.empty() || topology.vertices.empty() || topology.faces.empty() ||
        value.frames < 2U || value.vertices_per_frame != topology.vertices.size() ||
        value.positions.size() != value.frames * value.vertices_per_frame ||
        !std::isfinite(frames_per_second) || frames_per_second <= 0.0F) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "invalid topology, animation dimensions, output path, or frame rate"));
    }
    for (const auto & face : topology.faces) {
        if (face[0] >= topology.vertices.size() || face[1] >= topology.vertices.size() ||
            face[2] >= topology.vertices.size()) {
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "animation topology contains an out-of-range face"));
        }
    }
    for (const auto & position : value.positions) {
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "animation contains NaN or infinity"));
        }
    }

    std::vector<std::byte> binary;
    std::vector<binary_view> views;
    auto begin_view = [&] { align4(binary); return binary.size(); };
    auto end_view = [&](std::size_t begin) {
        views.push_back({begin, binary.size() - begin});
        return views.size() - 1U;
    };

    const auto positions_begin = begin_view();
    vec3 minimum = value.positions.front();
    vec3 maximum = minimum;
    for (std::size_t vertex = 0; vertex < value.vertices_per_frame; ++vertex) {
        const auto & position = value.positions[vertex];
        append_f32(binary, position.x); append_f32(binary, position.y); append_f32(binary, position.z);
        minimum.x = std::min(minimum.x, position.x); minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z); maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y); maximum.z = std::max(maximum.z, position.z);
    }
    const auto positions_view = end_view(positions_begin);

    const auto indices_begin = begin_view();
    for (const auto & face : topology.faces) {
        append_u32(binary, face[0]); append_u32(binary, face[1]); append_u32(binary, face[2]);
    }
    const auto indices_view = end_view(indices_begin);

    std::vector<std::size_t> target_views;
    target_views.reserve(value.frames - 1U);
    for (std::size_t frame = 1; frame < value.frames; ++frame) {
        const auto target_begin = begin_view();
        for (std::size_t vertex = 0; vertex < value.vertices_per_frame; ++vertex) {
            const auto & base = value.positions[vertex];
            const auto & position = value.positions[frame * value.vertices_per_frame + vertex];
            append_f32(binary, position.x - base.x);
            append_f32(binary, position.y - base.y);
            append_f32(binary, position.z - base.z);
        }
        target_views.push_back(end_view(target_begin));
    }

    const auto times_begin = begin_view();
    for (std::size_t frame = 0; frame < value.frames; ++frame) {
        append_f32(binary, static_cast<float>(frame) / frames_per_second);
    }
    const auto times_view = end_view(times_begin);
    const auto weights_begin = begin_view();
    const std::size_t targets = value.frames - 1U;
    for (std::size_t frame = 0; frame < value.frames; ++frame) {
        for (std::size_t target = 0; target < targets; ++target) {
            append_f32(binary, frame > 0U && target == frame - 1U ? 1.0F : 0.0F);
        }
    }
    const auto weights_view = end_view(weights_begin);
    align4(binary);

    nlohmann::json document;
    document["asset"] = {{"version", "2.0"}, {"generator", "animate-any-mesh.cpp"}};
    document["buffers"] = {{{"byteLength", binary.size()}}};
    document["bufferViews"] = nlohmann::json::array();
    for (std::size_t index = 0; index < views.size(); ++index) {
        nlohmann::json view{{"buffer", 0}, {"byteOffset", views[index].offset},
                            {"byteLength", views[index].length}};
        if (index == positions_view ||
            std::ranges::find(target_views, index) != target_views.end()) {
            view["target"] = 34962;
        } else if (index == indices_view) {
            view["target"] = 34963;
        }
        document["bufferViews"].push_back(std::move(view));
    }
    document["accessors"] = nlohmann::json::array();
    auto add_accessor = [&](std::size_t view, std::uint32_t component_type,
                            std::size_t count, const char * type) {
        const std::size_t index = document["accessors"].size();
        document["accessors"].push_back({{"bufferView", view},
                                          {"componentType", component_type},
                                          {"count", count}, {"type", type}});
        return index;
    };
    const auto position_accessor = add_accessor(positions_view, 5126U, value.vertices_per_frame, "VEC3");
    document["accessors"][position_accessor]["min"] = {minimum.x, minimum.y, minimum.z};
    document["accessors"][position_accessor]["max"] = {maximum.x, maximum.y, maximum.z};
    const auto index_accessor = add_accessor(indices_view, 5125U, topology.faces.size() * 3U, "SCALAR");
    std::vector<std::size_t> target_accessors;
    for (const auto target_view : target_views) {
        target_accessors.push_back(add_accessor(target_view, 5126U, value.vertices_per_frame, "VEC3"));
    }
    const auto time_accessor = add_accessor(times_view, 5126U, value.frames, "SCALAR");
    document["accessors"][time_accessor]["min"] = {0.0F};
    document["accessors"][time_accessor]["max"] = {
        static_cast<float>(value.frames - 1U) / frames_per_second};
    const auto weights_accessor = add_accessor(
        weights_view, 5126U, value.frames * targets, "SCALAR");

    nlohmann::json morph_targets = nlohmann::json::array();
    for (const auto accessor : target_accessors) morph_targets.push_back({{"POSITION", accessor}});
    document["meshes"] = {{{"primitives", {{{"attributes", {{"POSITION", position_accessor}}},
                                               {"indices", index_accessor},
                                               {"mode", 4}, {"targets", morph_targets}}}},
                             {"weights", std::vector<float>(targets, 0.0F)}}};
    document["nodes"] = {{{"mesh", 0}}};
    document["scenes"] = {{{"nodes", {0}}}};
    document["scene"] = 0;
    document["animations"] = {{{"samplers", {{{"input", time_accessor},
                                                 {"output", weights_accessor},
                                                 {"interpolation", "LINEAR"}}}},
                                {"channels", {{{"sampler", 0},
                                                {"target", {{"node", 0}, {"path", "weights"}}}}}}}};

    std::string json = document.dump();
    while ((json.size() % 4U) != 0U) json.push_back(' ');
    const std::size_t total_size = 12U + 8U + json.size() + 8U + binary.size();
    if (json.size() > std::numeric_limits<std::uint32_t>::max() ||
        binary.size() > std::numeric_limits<std::uint32_t>::max() ||
        total_size > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "animated GLB exceeds the 32-bit container limit"));
    }

    auto temporary = path;
    temporary += ".partial";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    auto write_u32 = [&](std::uint32_t number) {
        if constexpr (std::endian::native == std::endian::big) number = std::byteswap(number);
        stream.write(reinterpret_cast<const char *>(&number), sizeof(number));
    };
    write_u32(0x46546C67U); write_u32(2U); write_u32(static_cast<std::uint32_t>(total_size));
    write_u32(static_cast<std::uint32_t>(json.size())); write_u32(0x4E4F534AU);
    stream.write(json.data(), static_cast<std::streamsize>(json.size()));
    write_u32(static_cast<std::uint32_t>(binary.size())); write_u32(0x004E4942U);
    stream.write(reinterpret_cast<const char *>(binary.data()),
                 static_cast<std::streamsize>(binary.size()));
    stream.flush();
    if (!stream) {
        stream.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return std::unexpected(detail::fail(error_code::io, "failed to write animated GLB"));
    }
    stream.close();
    std::error_code filesystem_error;
    std::filesystem::rename(temporary, path, filesystem_error);
    if (filesystem_error) {
        const std::string publish_error = filesystem_error.message();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return std::unexpected(detail::fail(error_code::io,
            "failed to publish animated GLB: " + publish_error));
    }
    return {};
}

} // namespace aam
