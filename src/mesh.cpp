#include "internal.hpp"

#include <meshoptimizer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <latch>
#include <map>
#include <new>
#include <numeric>
#include <set>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>

namespace aam {
namespace {

using json = nlohmann::json;

constexpr std::uint32_t glb_magic = 0x46546C67U;
constexpr std::uint32_t json_chunk = 0x4E4F534AU;
constexpr std::uint32_t bin_chunk = 0x004E4942U;

[[nodiscard]] std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    return value;
}

template<class T>
[[nodiscard]] result<T> required_integer(const json & object, std::string_view key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_unsigned()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF field '" + std::string{key} + "' must be an unsigned integer"));
    }
    const auto value = it->get<std::uint64_t>();
    if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "glTF field '" + std::string{key} + "' is out of range"));
    }
    return static_cast<T>(value);
}

template<class T>
[[nodiscard]] result<T> optional_integer(const json & object, std::string_view key, T fallback) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    return required_integer<T>(object, key);
}

[[nodiscard]] result<const json *> array_item(
    const json & root, std::string_view array_name, std::size_t index) {
    const auto array_it = root.find(array_name);
    if (array_it == root.end() || !array_it->is_array() || index >= array_it->size()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF " + std::string{array_name} + " index is out of bounds"));
    }
    const auto & item = (*array_it)[index];
    if (!item.is_object()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF " + std::string{array_name} + " entry must be an object"));
    }
    return &item;
}

struct accessor_view {
    std::span<const std::byte> bytes;
    std::size_t count = 0;
    std::size_t stride = 0;
    std::uint32_t component_type = 0;
    std::string type;
};

[[nodiscard]] std::size_t component_size(std::uint32_t type) {
    switch (type) {
    case 5120U:
    case 5121U: return 1;
    case 5122U:
    case 5123U: return 2;
    case 5125U:
    case 5126U: return 4;
    default: return 0;
    }
}

[[nodiscard]] std::size_t component_count(std::string_view type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT2") return 4;
    if (type == "MAT3") return 9;
    if (type == "MAT4") return 16;
    return 0;
}

[[nodiscard]] result<accessor_view> get_accessor(
    const json & root, std::span<const std::byte> binary, std::size_t accessor_index) {
    auto accessor_result = array_item(root, "accessors", accessor_index);
    if (!accessor_result) return std::unexpected(accessor_result.error());
    const auto & accessor = **accessor_result;
    if (accessor.contains("sparse")) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "sparse glTF accessors are not supported"));
    }

    auto view_index = required_integer<std::size_t>(accessor, "bufferView");
    auto count = required_integer<std::size_t>(accessor, "count");
    auto type = required_integer<std::uint32_t>(accessor, "componentType");
    if (!view_index) return std::unexpected(view_index.error());
    if (!count) return std::unexpected(count.error());
    if (!type) return std::unexpected(type.error());
    const auto type_it = accessor.find("type");
    if (type_it == accessor.end() || !type_it->is_string()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF accessor type must be a string"));
    }
    const std::string shape = type_it->get<std::string>();
    const std::size_t scalar_bytes = component_size(*type);
    const std::size_t components = component_count(shape);
    if (scalar_bytes == 0 || components == 0) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "unsupported glTF accessor component or shape type"));
    }

    auto view_result = array_item(root, "bufferViews", *view_index);
    if (!view_result) return std::unexpected(view_result.error());
    const auto & view = **view_result;
    auto buffer_index = required_integer<std::size_t>(view, "buffer");
    auto view_length = required_integer<std::size_t>(view, "byteLength");
    auto view_offset = optional_integer<std::size_t>(view, "byteOffset", 0);
    auto accessor_offset = optional_integer<std::size_t>(accessor, "byteOffset", 0);
    auto stride = optional_integer<std::size_t>(view, "byteStride", scalar_bytes * components);
    if (!buffer_index) return std::unexpected(buffer_index.error());
    if (!view_length) return std::unexpected(view_length.error());
    if (!view_offset) return std::unexpected(view_offset.error());
    if (!accessor_offset) return std::unexpected(accessor_offset.error());
    if (!stride) return std::unexpected(stride.error());
    if (*buffer_index != 0) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "GLB may only reference its embedded buffer at index zero"));
    }

    std::size_t element_bytes = 0;
    if (!detail::checked_mul(scalar_bytes, components, element_bytes) ||
        *stride < element_bytes || *stride > 252U || (*stride % scalar_bytes) != 0U) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "invalid glTF accessor byte stride"));
    }
    if (*accessor_offset > *view_length) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF accessor starts outside its buffer view"));
    }

    std::size_t payload = 0;
    if (*count != 0) {
        std::size_t prefix = 0;
        if (!detail::checked_mul(*count - 1U, *stride, prefix) ||
            prefix > std::numeric_limits<std::size_t>::max() - element_bytes) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "glTF accessor size overflows address space"));
        }
        payload = prefix + element_bytes;
    }
    if (payload > *view_length - *accessor_offset ||
        *view_offset > binary.size() || *accessor_offset > binary.size() - *view_offset) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF accessor exceeds its buffer view"));
    }
    const std::size_t begin = *view_offset + *accessor_offset;
    if (payload > binary.size() - begin) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF accessor exceeds the GLB binary chunk"));
    }
    return accessor_view{binary.subspan(begin, payload), *count, *stride, *type, shape};
}

template<class T>
[[nodiscard]] T load_little(const std::byte * data) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    std::memcpy(&value, data, sizeof(value));
    if constexpr (std::endian::native == std::endian::big && std::is_integral_v<T>) {
        value = std::byteswap(value);
    }
    return value;
}

[[nodiscard]] result<std::uint32_t> read_index(const accessor_view & view, std::size_t i) {
    const auto * ptr = view.bytes.data() + i * view.stride;
    switch (view.component_type) {
    case 5121U: return std::to_integer<std::uint8_t>(*ptr);
    case 5123U: return load_little<std::uint16_t>(ptr);
    case 5125U: return load_little<std::uint32_t>(ptr);
    default:
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF triangle indices must be unsigned bytes, shorts, or integers"));
    }
}

[[nodiscard]] result<vec3> read_position(const accessor_view & view, std::size_t i) {
    if (view.component_type != 5126U || view.type != "VEC3") {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF POSITION must be a float VEC3 accessor"));
    }
    const auto * ptr = view.bytes.data() + i * view.stride;
    vec3 value{load_little<float>(ptr), load_little<float>(ptr + 4), load_little<float>(ptr + 8)};
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF POSITION contains NaN or infinity"));
    }
    return value;
}

using matrix4 = std::array<double, 16>;

[[nodiscard]] constexpr matrix4 identity_matrix() {
    return {1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0};
}

// glTF matrices are column-major and node transforms compose parent * local.
[[nodiscard]] matrix4 multiply(const matrix4 & left, const matrix4 & right) {
    matrix4 output{};
    for (std::size_t column = 0; column < 4U; ++column) {
        for (std::size_t row = 0; row < 4U; ++row) {
            for (std::size_t inner = 0; inner < 4U; ++inner) {
                output[column * 4U + row] +=
                    left[inner * 4U + row] * right[column * 4U + inner];
            }
        }
    }
    return output;
}

[[nodiscard]] result<std::vector<double>> numeric_array(
    const json & object, std::string_view key, std::size_t expected,
    std::span<const double> fallback) {
    const auto it = object.find(key);
    if (it == object.end()) return std::vector<double>(fallback.begin(), fallback.end());
    if (!it->is_array() || it->size() != expected) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF node '" + std::string{key} + "' must have " +
            std::to_string(expected) + " numeric elements"));
    }
    std::vector<double> values;
    values.reserve(expected);
    for (const auto & item : *it) {
        if (!item.is_number()) {
            return std::unexpected(detail::fail(error_code::invalid_format,
                "glTF node transform contains a non-numeric value"));
        }
        const double value = item.get<double>();
        if (!std::isfinite(value)) {
            return std::unexpected(detail::fail(error_code::invalid_format,
                "glTF node transform contains NaN or infinity"));
        }
        values.push_back(value);
    }
    return values;
}

[[nodiscard]] result<matrix4> local_transform(const json & node) {
    const bool has_matrix = node.contains("matrix");
    const bool has_trs = node.contains("translation") || node.contains("rotation") ||
                         node.contains("scale");
    if (has_matrix && has_trs) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF node may not define both matrix and TRS transforms"));
    }
    if (has_matrix) {
        constexpr std::array<double, 16> identity = identity_matrix();
        auto values = numeric_array(node, "matrix", 16U, identity);
        if (!values) return std::unexpected(values.error());
        matrix4 output{};
        std::ranges::copy(*values, output.begin());
        return output;
    }

    constexpr std::array<double, 3> zero{0.0, 0.0, 0.0};
    constexpr std::array<double, 3> one{1.0, 1.0, 1.0};
    constexpr std::array<double, 4> rotation_identity{0.0, 0.0, 0.0, 1.0};
    auto translation = numeric_array(node, "translation", 3U, zero);
    auto rotation = numeric_array(node, "rotation", 4U, rotation_identity);
    auto scale = numeric_array(node, "scale", 3U, one);
    if (!translation) return std::unexpected(translation.error());
    if (!rotation) return std::unexpected(rotation.error());
    if (!scale) return std::unexpected(scale.error());

    double x = (*rotation)[0];
    double y = (*rotation)[1];
    double z = (*rotation)[2];
    double w = (*rotation)[3];
    const double norm = std::sqrt(x * x + y * y + z * z + w * w);
    if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF node rotation quaternion has zero length"));
    }
    x /= norm; y /= norm; z /= norm; w /= norm;
    const double sx = (*scale)[0];
    const double sy = (*scale)[1];
    const double sz = (*scale)[2];
    return matrix4{
        (1.0 - 2.0 * (y * y + z * z)) * sx,
        (2.0 * (x * y + z * w)) * sx,
        (2.0 * (x * z - y * w)) * sx, 0.0,
        (2.0 * (x * y - z * w)) * sy,
        (1.0 - 2.0 * (x * x + z * z)) * sy,
        (2.0 * (y * z + x * w)) * sy, 0.0,
        (2.0 * (x * z + y * w)) * sz,
        (2.0 * (y * z - x * w)) * sz,
        (1.0 - 2.0 * (x * x + y * y)) * sz, 0.0,
        (*translation)[0], (*translation)[1], (*translation)[2], 1.0};
}

[[nodiscard]] result<vec3> transform_position(const matrix4 & matrix, const vec3 & input) {
    const double x = matrix[0] * input.x + matrix[4] * input.y +
                     matrix[8] * input.z + matrix[12];
    const double y = matrix[1] * input.x + matrix[5] * input.y +
                     matrix[9] * input.z + matrix[13];
    const double z = matrix[2] * input.x + matrix[6] * input.y +
                     matrix[10] * input.z + matrix[14];
    constexpr double maximum = std::numeric_limits<float>::max();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        std::abs(x) > maximum || std::abs(y) > maximum || std::abs(z) > maximum) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "glTF world-space position is outside the finite float range"));
    }
    return vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

} // namespace

result<mesh> load_glb(std::span<const std::byte> bytes, const mesh_limits & limits) {
    if (bytes.size() < 12U) {
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB header is truncated"));
    }
    if (bytes.size() > limits.max_file_bytes) {
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB exceeds configured file-size limit"));
    }
    if (read_u32(bytes, 0) != glb_magic || read_u32(bytes, 4) != 2U) {
        return std::unexpected(detail::fail(error_code::invalid_format, "expected a version-2 binary glTF file"));
    }
    const auto declared_size = static_cast<std::size_t>(read_u32(bytes, 8));
    if (declared_size != bytes.size()) {
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB declared length does not match input length"));
    }

    std::span<const std::byte> json_bytes;
    std::span<const std::byte> binary;
    std::size_t cursor = 12U;
    while (cursor < bytes.size()) {
        if (bytes.size() - cursor < 8U) {
            return std::unexpected(detail::fail(error_code::invalid_format, "GLB chunk header is truncated"));
        }
        const auto length = static_cast<std::size_t>(read_u32(bytes, cursor));
        const auto type = read_u32(bytes, cursor + 4U);
        cursor += 8U;
        if (length > bytes.size() - cursor) {
            return std::unexpected(detail::fail(error_code::invalid_format, "GLB chunk payload is truncated"));
        }
        const auto payload = bytes.subspan(cursor, length);
        if (type == json_chunk && json_bytes.empty()) {
            json_bytes = payload;
        } else if (type == bin_chunk && binary.empty()) {
            binary = payload;
        }
        cursor += length;
    }
    if (json_bytes.empty() || binary.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB requires JSON and BIN chunks"));
    }
    if (json_bytes.size() > limits.max_json_bytes) {
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB JSON exceeds configured limit"));
    }

    const auto * json_begin = reinterpret_cast<const char *>(json_bytes.data());
    auto root = json::parse(json_begin, json_begin + json_bytes.size(), nullptr, false, true);
    if (root.is_discarded() || !root.is_object()) {
        return std::unexpected(detail::fail(error_code::invalid_format, "GLB contains invalid JSON"));
    }
    const auto meshes_it = root.find("meshes");
    if (meshes_it == root.end() || !meshes_it->is_array()) {
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF has no meshes array"));
    }

    std::vector<mesh> mesh_assets;
    mesh_assets.reserve(meshes_it->size());
    std::size_t parsed_vertices = 0U;
    std::size_t parsed_faces = 0U;
    for (const auto & mesh_json : *meshes_it) {
        if (!mesh_json.is_object()) {
            return std::unexpected(detail::fail(error_code::invalid_format,
                "glTF mesh entry must be an object"));
        }
        mesh asset;
        const auto primitives_it = mesh_json.find("primitives");
        if (primitives_it == mesh_json.end() || !primitives_it->is_array()) {
            return std::unexpected(detail::fail(error_code::invalid_format, "glTF mesh has no primitives array"));
        }
        for (const auto & primitive : *primitives_it) {
            if (!primitive.is_object()) {
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF primitive must be an object"));
            }
            auto mode = optional_integer<std::uint32_t>(primitive, "mode", 4U);
            if (!mode) return std::unexpected(mode.error());
            if (*mode != 4U) {
                continue;
            }
            const auto attributes_it = primitive.find("attributes");
            if (attributes_it == primitive.end() || !attributes_it->is_object()) {
                return std::unexpected(detail::fail(error_code::invalid_format, "glTF primitive has no attributes"));
            }
            auto position_index = required_integer<std::size_t>(*attributes_it, "POSITION");
            if (!position_index) return std::unexpected(position_index.error());
            auto positions = get_accessor(root, binary, *position_index);
            if (!positions) return std::unexpected(positions.error());
            if (parsed_vertices > limits.max_vertices ||
                positions->count > limits.max_vertices - parsed_vertices) {
                return std::unexpected(detail::fail(error_code::limit_exceeded, "glTF exceeds configured vertex limit"));
            }
            parsed_vertices += positions->count;
            const std::size_t base = asset.vertices.size();
            asset.vertices.reserve(base + positions->count);
            for (std::size_t i = 0; i < positions->count; ++i) {
                auto position = read_position(*positions, i);
                if (!position) return std::unexpected(position.error());
                asset.vertices.push_back(*position);
            }

            std::vector<std::uint32_t> indices;
            const auto index_it = primitive.find("indices");
            if (index_it == primitive.end()) {
                if ((positions->count % 3U) != 0U) {
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "non-indexed triangle primitive vertex count is not divisible by three"));
                }
                indices.resize(positions->count);
                std::iota(indices.begin(), indices.end(), 0U);
            } else {
                auto index_number = required_integer<std::size_t>(primitive, "indices");
                if (!index_number) return std::unexpected(index_number.error());
                auto accessor = get_accessor(root, binary, *index_number);
                if (!accessor) return std::unexpected(accessor.error());
                if (accessor->type != "SCALAR" || (accessor->count % 3U) != 0U) {
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "glTF triangle index accessor must be SCALAR and divisible by three"));
                }
                indices.reserve(accessor->count);
                for (std::size_t i = 0; i < accessor->count; ++i) {
                    auto index = read_index(*accessor, i);
                    if (!index) return std::unexpected(index.error());
                    if (*index >= positions->count) {
                        return std::unexpected(detail::fail(error_code::invalid_format,
                            "glTF triangle index is out of bounds"));
                    }
                    indices.push_back(*index);
                }
            }
            const std::size_t face_count = indices.size() / 3U;
            if (parsed_faces > limits.max_faces ||
                face_count > limits.max_faces - parsed_faces) {
                return std::unexpected(detail::fail(error_code::limit_exceeded, "glTF exceeds configured face limit"));
            }
            parsed_faces += face_count;
            asset.faces.reserve(asset.faces.size() + face_count);
            for (std::size_t i = 0; i < indices.size(); i += 3U) {
                const auto a = base + indices[i];
                const auto b = base + indices[i + 1U];
                const auto c = base + indices[i + 2U];
                if (a > std::numeric_limits<std::uint32_t>::max() ||
                    b > std::numeric_limits<std::uint32_t>::max() ||
                    c > std::numeric_limits<std::uint32_t>::max()) {
                    return std::unexpected(detail::fail(error_code::limit_exceeded, "glTF vertex index exceeds 32 bits"));
                }
                asset.faces.push_back({static_cast<std::uint32_t>(a),
                                       static_cast<std::uint32_t>(b),
                                       static_cast<std::uint32_t>(c)});
            }
        }
        mesh_assets.push_back(std::move(asset));
    }

    mesh output;
    auto append_asset = [&](std::size_t asset_index, const matrix4 & transform) -> result<void> {
        if (asset_index >= mesh_assets.size()) {
            return std::unexpected(detail::fail(error_code::invalid_format,
                "glTF node mesh index is out of bounds"));
        }
        const auto & asset = mesh_assets[asset_index];
        if (output.vertices.size() > limits.max_vertices ||
            asset.vertices.size() > limits.max_vertices - output.vertices.size() ||
            output.faces.size() > limits.max_faces ||
            asset.faces.size() > limits.max_faces - output.faces.size()) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "instantiated glTF scene exceeds configured mesh limits"));
        }
        const std::size_t base = output.vertices.size();
        output.vertices.reserve(base + asset.vertices.size());
        for (const auto & vertex : asset.vertices) {
            auto transformed = transform_position(transform, vertex);
            if (!transformed) return std::unexpected(transformed.error());
            output.vertices.push_back(*transformed);
        }
        output.faces.reserve(output.faces.size() + asset.faces.size());
        for (const auto & face : asset.faces) {
            const std::size_t a = base + face[0];
            const std::size_t b = base + face[1];
            const std::size_t c = base + face[2];
            if (a > std::numeric_limits<std::uint32_t>::max() ||
                b > std::numeric_limits<std::uint32_t>::max() ||
                c > std::numeric_limits<std::uint32_t>::max()) {
                return std::unexpected(detail::fail(error_code::limit_exceeded,
                    "instantiated glTF vertex index exceeds 32 bits"));
            }
            output.faces.push_back({static_cast<std::uint32_t>(a),
                                    static_cast<std::uint32_t>(b),
                                    static_cast<std::uint32_t>(c)});
        }
        return {};
    };

    const auto nodes_it = root.find("nodes");
    if (nodes_it != root.end() && !nodes_it->is_array()) {
        return std::unexpected(detail::fail(error_code::invalid_format,
            "glTF nodes field must be an array"));
    }
    if (nodes_it == root.end() || nodes_it->empty()) {
        for (std::size_t index = 0; index < mesh_assets.size(); ++index) {
            auto appended = append_asset(index, identity_matrix());
            if (!appended) return std::unexpected(appended.error());
        }
    } else {
        std::vector<std::size_t> roots;
        const auto scenes_it = root.find("scenes");
        if (scenes_it != root.end()) {
            if (!scenes_it->is_array() || scenes_it->empty()) {
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "glTF scenes field must be a non-empty array"));
            }
            auto scene_index = optional_integer<std::size_t>(root, "scene", 0U);
            if (!scene_index) return std::unexpected(scene_index.error());
            auto scene = array_item(root, "scenes", *scene_index);
            if (!scene) return std::unexpected(scene.error());
            const auto scene_nodes = (**scene).find("nodes");
            if (scene_nodes != (**scene).end()) {
                if (!scene_nodes->is_array()) {
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "glTF scene nodes field must be an array"));
                }
                for (const auto & value : *scene_nodes) {
                    if (!value.is_number_unsigned()) {
                        return std::unexpected(detail::fail(error_code::invalid_format,
                            "glTF scene node index must be unsigned"));
                    }
                    roots.push_back(value.get<std::size_t>());
                }
            }
        } else {
            // For a document without a selected scene, flatten every root node.
            std::vector<bool> child(nodes_it->size(), false);
            for (const auto & node : *nodes_it) {
                if (!node.is_object()) continue;
                const auto children = node.find("children");
                if (children == node.end()) continue;
                if (!children->is_array()) {
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "glTF node children field must be an array"));
                }
                for (const auto & value : *children) {
                    if (!value.is_number_unsigned() || value.get<std::uint64_t>() >= child.size()) {
                        return std::unexpected(detail::fail(error_code::invalid_format,
                            "glTF child node index is out of bounds"));
                    }
                    child[static_cast<std::size_t>(value.get<std::uint64_t>())] = true;
                }
            }
            for (std::size_t index = 0; index < child.size(); ++index) {
                if (!child[index]) roots.push_back(index);
            }
        }

        struct pending_node { std::size_t index; matrix4 parent; };
        std::vector<pending_node> pending;
        pending.reserve(nodes_it->size());
        for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
            pending.push_back({*it, identity_matrix()});
        }
        std::vector<bool> visited(nodes_it->size(), false);
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            if (current.index >= nodes_it->size()) {
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "glTF scene node index is out of bounds"));
            }
            if (visited[current.index]) {
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "glTF node graph contains a cycle or multiple parents"));
            }
            visited[current.index] = true;
            const auto & node = (*nodes_it)[current.index];
            if (!node.is_object()) {
                return std::unexpected(detail::fail(error_code::invalid_format,
                    "glTF node entry must be an object"));
            }
            auto local = local_transform(node);
            if (!local) return std::unexpected(local.error());
            const matrix4 world = multiply(current.parent, *local);
            if (node.contains("mesh")) {
                auto mesh_index = required_integer<std::size_t>(node, "mesh");
                if (!mesh_index) return std::unexpected(mesh_index.error());
                auto appended = append_asset(*mesh_index, world);
                if (!appended) return std::unexpected(appended.error());
            }
            const auto children = node.find("children");
            if (children != node.end()) {
                if (!children->is_array()) {
                    return std::unexpected(detail::fail(error_code::invalid_format,
                        "glTF node children field must be an array"));
                }
                for (auto it = children->rbegin(); it != children->rend(); ++it) {
                    if (!it->is_number_unsigned()) {
                        return std::unexpected(detail::fail(error_code::invalid_format,
                            "glTF child node index must be unsigned"));
                    }
                    pending.push_back({it->get<std::size_t>(), world});
                }
            }
        }
    }
    if (output.vertices.empty() || output.faces.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_format, "glTF contains no triangle mesh data"));
    }
    return output;
}

result<mesh> load_glb_file(const std::filesystem::path & path, const mesh_limits & limits) {
    std::error_code filesystem_error;
    const auto file_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        return std::unexpected(detail::fail(error_code::io,
            "cannot stat input GLB: " + filesystem_error.message()));
    }
    if (file_size > limits.max_file_bytes || file_size > std::numeric_limits<std::size_t>::max()) {
        return std::unexpected(detail::fail(error_code::limit_exceeded, "GLB exceeds configured file-size limit"));
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::unexpected(detail::fail(error_code::io, "cannot open input GLB"));
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    if (!bytes.empty() && !stream.read(reinterpret_cast<char *>(bytes.data()),
                                       static_cast<std::streamsize>(bytes.size()))) {
        return std::unexpected(detail::fail(error_code::io, "failed while reading input GLB"));
    }
    return load_glb(bytes, limits);
}

namespace {

[[nodiscard]] result<void> validate_simplification_input(
    const mesh & source, std::size_t max_vertices) {
    if (max_vertices < 3U) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "maximum vertex count must be at least three"));
    }
    if (source.vertices.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh has no vertices"));
    }
    if (source.vertices.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "mesh has too many vertices for 32-bit triangle indices"));
    }
    for (const auto & value : source.vertices) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "mesh contains NaN or infinity"));
        }
    }
    for (const auto & face : source.faces) {
        if (face[0] >= source.vertices.size() || face[1] >= source.vertices.size() ||
            face[2] >= source.vertices.size()) {
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "mesh face index is out of bounds"));
        }
    }
    return {};
}

[[nodiscard]] result<mesh> compact_indexed_mesh(
    std::span<const vec3> vertices, std::span<const std::uint32_t> indices) {
    if (indices.empty() || indices.size() % 3U != 0U) {
        return std::unexpected(detail::fail(error_code::compute,
            "mesh simplifier produced an invalid index buffer"));
    }
    constexpr auto unused = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> remap(vertices.size(), unused);
    mesh output;
    output.vertices.reserve(std::min(vertices.size(), indices.size()));
    output.faces.reserve(indices.size() / 3U);
    for (std::size_t offset = 0; offset < indices.size(); offset += 3U) {
        triangle face{};
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const auto source_index = indices[offset + corner];
            if (source_index >= vertices.size()) {
                return std::unexpected(detail::fail(error_code::compute,
                    "mesh simplifier returned an out-of-bounds index"));
            }
            auto & destination_index = remap[source_index];
            if (destination_index == unused) {
                if (output.vertices.size() >= unused) {
                    return std::unexpected(detail::fail(error_code::limit_exceeded,
                        "simplified mesh exceeds 32-bit triangle indices"));
                }
                destination_index = static_cast<std::uint32_t>(output.vertices.size());
                output.vertices.push_back(vertices[source_index]);
            }
            face[corner] = destination_index;
        }
        if (face[0] != face[1] && face[1] != face[2] && face[0] != face[2]) {
            output.faces.push_back(face);
        }
    }
    if (output.faces.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "mesh has no non-degenerate triangle faces"));
    }
    return output;
}

[[nodiscard]] std::size_t aligned_triangle_index_target(
    std::size_t vertex_budget, std::size_t index_count) noexcept {
    std::size_t target = index_count;
    std::size_t scaled = 0U;
    if (detail::checked_mul(vertex_budget, std::size_t{6U}, scaled)) {
        target = std::min(target, scaled);
    }
    target -= target % 3U;
    return std::max(target, std::size_t{3U});
}

[[nodiscard]] std::size_t tighter_index_target(
    std::size_t current_target, std::size_t result_indices,
    std::size_t result_vertices, std::size_t vertex_budget) noexcept {
    std::size_t product = 0U;
    std::size_t next = current_target / 2U;
    if (detail::checked_mul(result_indices, vertex_budget, product) && result_vertices != 0U) {
        next = product / result_vertices;
        next -= next / 10U;
    }
    next -= next % 3U;
    if (next >= current_target) next = current_target > 3U ? current_target - 3U : 3U;
    return std::max(next, std::size_t{3U});
}

[[nodiscard]] result<mesh> bounded_triangle_subset(
    std::span<const vec3> vertices, std::span<const std::uint32_t> indices,
    std::size_t max_vertices) {
    std::vector<std::uint8_t> referenced(vertices.size(), 0U);
    std::size_t referenced_count = 0U;
    std::vector<std::uint32_t> bounded;
    std::size_t reserve_count = indices.size();
    std::size_t scaled_budget = 0U;
    if (detail::checked_mul(max_vertices, std::size_t{3U}, scaled_budget)) {
        reserve_count = std::min(reserve_count, scaled_budget);
    }
    bounded.reserve(reserve_count);
    for (std::size_t offset = 0; offset + 2U < indices.size(); offset += 3U) {
        std::size_t additions = 0U;
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            additions += referenced[indices[offset + corner]] == 0U ? 1U : 0U;
        }
        if (referenced_count + additions > max_vertices) continue;
        for (std::size_t corner = 0; corner < 3U; ++corner) {
            const auto index = indices[offset + corner];
            if (referenced[index] == 0U) {
                referenced[index] = 1U;
                ++referenced_count;
            }
            bounded.push_back(index);
        }
    }
    return compact_indexed_mesh(vertices, bounded);
}

} // namespace

result<mesh> simplify_mesh(const mesh & source, std::size_t max_vertices) {
    try {
        auto valid = validate_simplification_input(source, max_vertices);
        if (!valid) return std::unexpected(valid.error());
        if (source.faces.empty()) {
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "mesh has no triangle faces"));
        }
        if (source.vertices.size() <= max_vertices) return source;

        // Merge exact positions before simplification. Besides matching
        // prepare_mesh, this removes geometric degenerates that violate the
        // simplifier's preconditions.
        using key = std::tuple<float, float, float>;
        std::map<key, std::uint32_t> unique_vertices;
        for (const auto & value : source.vertices) {
            unique_vertices.try_emplace({value.x, value.y, value.z}, 0U);
        }
        std::vector<vec3> vertices;
        vertices.reserve(unique_vertices.size());
        std::uint32_t next_vertex = 0U;
        for (auto & [value, index] : unique_vertices) {
            index = next_vertex++;
            vertices.push_back({std::get<0>(value), std::get<1>(value), std::get<2>(value)});
        }
        std::vector<std::uint32_t> original_to_unique;
        original_to_unique.reserve(source.vertices.size());
        for (const auto & value : source.vertices) {
            original_to_unique.push_back(unique_vertices.at({value.x, value.y, value.z}));
        }

        std::set<triangle> face_keys;
        std::vector<std::uint32_t> indices;
        std::size_t index_capacity = 0U;
        if (!detail::checked_mul(source.faces.size(), std::size_t{3U}, index_capacity)) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "mesh index count overflows address space"));
        }
        indices.reserve(index_capacity);
        for (const auto & source_face : source.faces) {
            triangle face{original_to_unique[source_face[0]],
                          original_to_unique[source_face[1]],
                          original_to_unique[source_face[2]]};
            triangle face_key = face;
            std::ranges::sort(face_key);
            if (face_key[0] == face_key[1] || face_key[1] == face_key[2] ||
                !face_keys.insert(face_key).second) {
                continue;
            }
            indices.insert(indices.end(), face.begin(), face.end());
        }
        auto canonical = compact_indexed_mesh(vertices, indices);
        if (!canonical) return std::unexpected(canonical.error());
        if (canonical->vertices.size() <= max_vertices) return canonical;

        vertices = std::move(canonical->vertices);
        indices.clear();
        std::size_t canonical_index_count = 0U;
        if (!detail::checked_mul(canonical->faces.size(), std::size_t{3U},
                                 canonical_index_count)) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "canonical mesh index count overflows address space"));
        }
        indices.reserve(canonical_index_count);
        for (const auto & face : canonical->faces) {
            indices.insert(indices.end(), face.begin(), face.end());
        }

        static_assert(sizeof(vec3) == 3U * sizeof(float));
        static_assert(std::is_standard_layout_v<vec3>);
        std::vector<std::uint32_t> simplified(indices.size());
        std::vector<std::uint32_t> best = indices;
        std::size_t best_vertices = vertices.size();
        auto target = aligned_triangle_index_target(max_vertices, indices.size());

        constexpr std::size_t maximum_attempts = 8U;
        for (std::size_t attempt = 0U; attempt < maximum_attempts; ++attempt) {
            float result_error = 0.0F;
            const auto result_count = meshopt_simplify(
                simplified.data(), indices.data(), indices.size(),
                &vertices.front().x, vertices.size(), sizeof(vec3), target, 1.0F,
                meshopt_SimplifyPermissive | meshopt_SimplifyRegularize,
                &result_error);
            if (result_count == 0U || result_count > indices.size() || result_count % 3U != 0U) {
                return std::unexpected(detail::fail(error_code::compute,
                    "mesh simplifier returned an invalid index count"));
            }
            simplified.resize(result_count);
            auto compact = compact_indexed_mesh(vertices, simplified);
            if (!compact) return std::unexpected(compact.error());
            if (compact->vertices.size() < best_vertices) {
                best_vertices = compact->vertices.size();
                best.assign(simplified.begin(), simplified.end());
            }
            if (compact->vertices.size() <= max_vertices) return compact;
            target = tighter_index_target(
                target, result_count, compact->vertices.size(), max_vertices);
            simplified.resize(indices.size());
        }

        // Sloppy simplification crosses topology boundaries when QEM cannot
        // meet the hard budget (for example, thousands of disconnected parts).
        target = aligned_triangle_index_target(max_vertices, indices.size());
        for (std::size_t attempt = 0U; attempt < maximum_attempts; ++attempt) {
            float result_error = 0.0F;
            const auto result_count = meshopt_simplifySloppy(
                simplified.data(), indices.data(), indices.size(),
                &vertices.front().x, vertices.size(), sizeof(vec3), nullptr,
                target, 1.0F, &result_error);
            if (result_count == 0U || result_count > indices.size() || result_count % 3U != 0U) {
                return std::unexpected(detail::fail(error_code::compute,
                    "sloppy mesh simplifier returned an invalid index count"));
            }
            simplified.resize(result_count);
            auto compact = compact_indexed_mesh(vertices, simplified);
            if (!compact) return std::unexpected(compact.error());
            if (compact->vertices.size() < best_vertices) {
                best_vertices = compact->vertices.size();
                best.assign(simplified.begin(), simplified.end());
            }
            if (compact->vertices.size() <= max_vertices) return compact;
            target = tighter_index_target(
                target, result_count, compact->vertices.size(), max_vertices);
            simplified.resize(indices.size());
        }

        // The public contract is a strict memory bound. Retain as much of the
        // best simplifier result as possible if pathological topology prevented
        // both algorithms from reaching it naturally.
        return bounded_triangle_subset(vertices, best, max_vertices);
    } catch (const std::bad_alloc &) {
        return std::unexpected(detail::fail(error_code::allocation,
            "not enough memory to simplify mesh"));
    } catch (const std::length_error &) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "mesh simplification exceeds container limits"));
    }
}

result<prepared_mesh> prepare_mesh(const mesh & source, std::size_t max_vertices) {
    if (source.vertices.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh has no vertices"));
    }
    if (max_vertices == 0U) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "maximum vertex count must be positive"));
    }
    for (const auto & value : source.vertices) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh contains NaN or infinity"));
        }
    }
    for (const auto & face : source.faces) {
        if (face[0] >= source.vertices.size() || face[1] >= source.vertices.size() ||
            face[2] >= source.vertices.size()) {
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh face index is out of bounds"));
        }
    }

    // torch.unique(..., dim=0, sorted=True) in the reference produces a
    // lexicographically sorted vertex array and an inverse index. std::map
    // deliberately gives +0 and -0 the same key, matching float equality.
    using key = std::tuple<float, float, float>;
    std::map<key, std::uint32_t> sorted_vertices;
    for (const auto & value : source.vertices) {
        sorted_vertices.try_emplace({value.x, value.y, value.z}, 0U);
    }
    if (sorted_vertices.size() > max_vertices) {
        return std::unexpected(detail::fail(error_code::limit_exceeded,
            "merged mesh has " + std::to_string(sorted_vertices.size()) +
            " vertices, exceeding the configured limit of " + std::to_string(max_vertices)));
    }

    prepared_mesh output;
    output.value.vertices.reserve(sorted_vertices.size());
    std::uint32_t next = 0;
    for (auto & [value, index] : sorted_vertices) {
        index = next++;
        output.value.vertices.push_back({std::get<0>(value), std::get<1>(value), std::get<2>(value)});
    }
    output.original_to_merged.reserve(source.vertices.size());
    for (const auto & value : source.vertices) {
        output.original_to_merged.push_back(sorted_vertices.at({value.x, value.y, value.z}));
    }

    std::set<triangle> unique_faces;
    for (const auto & face : source.faces) {
        triangle mapped{output.original_to_merged[face[0]],
                        output.original_to_merged[face[1]],
                        output.original_to_merged[face[2]]};
        std::ranges::sort(mapped);
        if (mapped[0] != mapped[1] && mapped[1] != mapped[2]) {
            unique_faces.insert(mapped);
        }
    }
    output.value.faces.assign(unique_faces.begin(), unique_faces.end());
    if (output.value.faces.empty()) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "mesh has no non-degenerate triangle faces after vertex merging"));
    }

    vec3 low = output.value.vertices.front();
    vec3 high = low;
    for (const auto & value : output.value.vertices) {
        low.x = std::min(low.x, value.x); low.y = std::min(low.y, value.y); low.z = std::min(low.z, value.z);
        high.x = std::max(high.x, value.x); high.y = std::max(high.y, value.y); high.z = std::max(high.z, value.z);
    }
    output.transform.center = {(low.x + high.x) * 0.5F,
                               (low.y + high.y) * 0.5F,
                               (low.z + high.z) * 0.5F};
    float scale = 0.0F;
    for (auto & value : output.value.vertices) {
        value.x -= output.transform.center.x;
        value.y -= output.transform.center.y;
        value.z -= output.transform.center.z;
        scale = std::max({scale, std::abs(value.x), std::abs(value.y), std::abs(value.z)});
    }
    output.transform.scale = scale + 1.0e-8F;
    for (auto & value : output.value.vertices) {
        value.x /= output.transform.scale;
        value.y /= output.transform.scale;
        value.z /= output.transform.scale;
    }

    std::vector<std::vector<std::uint32_t>> neighbors(output.value.vertices.size());
    for (const auto & face : output.value.faces) {
        for (std::size_t edge = 0; edge < 3U; ++edge) {
            const auto a = face[edge];
            const auto b = face[(edge + 1U) % 3U];
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
    }
    output.adjacency_offsets.reserve(neighbors.size() + 1U);
    output.adjacency_offsets.push_back(0U);
    for (auto & row : neighbors) {
        std::ranges::sort(row);
        row.erase(std::unique(row.begin(), row.end()), row.end());
        if (row.size() > std::numeric_limits<std::uint32_t>::max() - output.adjacency_indices.size()) {
            return std::unexpected(detail::fail(error_code::limit_exceeded,
                "mesh adjacency exceeds 32-bit storage"));
        }
        output.adjacency_indices.insert(output.adjacency_indices.end(), row.begin(), row.end());
        output.adjacency_offsets.push_back(static_cast<std::uint32_t>(output.adjacency_indices.size()));
    }
    return output;
}

result<std::vector<std::uint32_t>> farthest_point_sample(
    std::span<const float> points,
    std::size_t point_count,
    std::size_t dimensions,
    std::size_t sample_count) {
    if (point_count == 0U ||
        point_count > std::numeric_limits<std::uint32_t>::max() ||
        dimensions == 0U || sample_count == 0U || sample_count > point_count) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "FPS dimensions must be positive, points must fit in uint32, and samples may not exceed points"));
    }
    std::size_t expected = 0;
    if (!detail::checked_mul(point_count, dimensions, expected) || points.size() != expected) {
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "FPS input span does not match point_count * dimensions"));
    }
    if (!std::ranges::all_of(points, [](float value) { return std::isfinite(value); })) {
        return std::unexpected(detail::fail(error_code::invalid_argument, "FPS input contains NaN or infinity"));
    }

    std::vector<std::uint32_t> selected;
    selected.reserve(sample_count);
    selected.push_back(0U); // PyTorch3D random_start_point=False
    std::vector<float> closest(point_count, std::numeric_limits<float>::infinity());

    // Each FPS iteration depends on the preceding selected point, but the
    // distance update for every candidate point is independent. Keep a fixed
    // worker team alive across all iterations so large embeddings do not pay
    // thread-creation or parallel-region overhead 1,000+ times. The ordered
    // reduction preserves the serial implementation's lowest-index tie break.
    const auto hardware_threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
    const std::size_t desired_workers = std::min(
        hardware_threads, std::max<std::size_t>(1U, point_count / 128U));
    constexpr std::size_t parallel_work_threshold = 1U << 20U;
    const bool enough_work = dimensions <=
        std::numeric_limits<std::size_t>::max() / point_count &&
        point_count * dimensions >= parallel_work_threshold;
    if (desired_workers > 1U && sample_count > 2U && enough_work) {
        struct alignas(64) local_maximum {
            std::size_t index = 0U;
            float distance = -1.0F;
        };
        std::vector<local_maximum> local_maxima(desired_workers);
        std::size_t active_workers = 0U;
        auto select_farthest = [&]() noexcept {
            std::size_t farthest = 0U;
            float farthest_distance = -1.0F;
            for (std::size_t worker = 0; worker < active_workers; ++worker) {
                const auto & candidate = local_maxima[worker];
                if (candidate.distance > farthest_distance) {
                    farthest = candidate.index;
                    farthest_distance = candidate.distance;
                }
            }
            selected.push_back(static_cast<std::uint32_t>(farthest));
        };
        using phase_barrier = std::barrier<decltype(select_farthest)>;
        phase_barrier * phase = nullptr;
        std::latch start_gate{1};
        std::vector<std::jthread> workers;
        workers.reserve(desired_workers);
        for (std::size_t worker = 0; worker < desired_workers; ++worker) {
            try {
                workers.emplace_back([&, worker] {
                    start_gate.wait();
                    if (phase == nullptr) return;
                    const std::size_t points_per_worker = point_count / active_workers;
                    const std::size_t remainder = point_count % active_workers;
                    const std::size_t begin = worker * points_per_worker +
                                              std::min(worker, remainder);
                    const std::size_t end = begin + points_per_worker +
                                            static_cast<std::size_t>(worker < remainder);
                    for (std::size_t iteration = 1U; iteration < sample_count; ++iteration) {
                        const auto latest = static_cast<std::size_t>(selected.back());
                        const auto latest_point = points.subspan(latest * dimensions, dimensions);
                        std::size_t farthest = begin;
                        float farthest_distance = -1.0F;
                        for (std::size_t i = begin; i < end; ++i) {
                            float distance = 0.0F;
                            const auto candidate = points.subspan(i * dimensions, dimensions);
                            for (std::size_t d = 0; d < dimensions; ++d) {
                                const float delta = candidate[d] - latest_point[d];
                                distance = std::fma(delta, delta, distance);
                            }
                            closest[i] = std::min(closest[i], distance);
                            if (closest[i] > farthest_distance) {
                                farthest_distance = closest[i];
                                farthest = i;
                            }
                        }
                        local_maxima[worker] = {farthest, farthest_distance};
                        phase->arrive_and_wait();
                    }
                });
            } catch (const std::system_error &) {
                break;
            }
        }
        active_workers = workers.size();
        if (active_workers != 0U) {
            phase_barrier iterations{
                static_cast<std::ptrdiff_t>(active_workers), select_farthest};
            phase = &iterations;
            start_gate.count_down();
            workers.clear(); // std::jthread joins before the barrier leaves scope.
            return selected;
        }
        start_gate.count_down();
    }

    for (std::size_t iteration = 1U; iteration < sample_count; ++iteration) {
        const auto latest = static_cast<std::size_t>(selected.back());
        const auto latest_point = points.subspan(latest * dimensions, dimensions);
        std::size_t farthest = 0U;
        float farthest_distance = -1.0F;
        for (std::size_t i = 0; i < point_count; ++i) {
            float distance = 0.0F;
            const auto candidate = points.subspan(i * dimensions, dimensions);
            for (std::size_t d = 0; d < dimensions; ++d) {
                const float delta = candidate[d] - latest_point[d];
                distance = std::fma(delta, delta, distance);
            }
            closest[i] = std::min(closest[i], distance);
            if (closest[i] > farthest_distance) {
                farthest_distance = closest[i];
                farthest = i;
            }
        }
        selected.push_back(static_cast<std::uint32_t>(farthest));
    }
    return selected;
}

} // namespace aam
