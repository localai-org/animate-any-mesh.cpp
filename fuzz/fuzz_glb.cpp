#include "aam/aam.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    if (size == 0U || size > (16U << 20U)) return 0;
    aam::mesh_limits limits;
    limits.max_file_bytes = 16U << 20U;
    limits.max_json_bytes = 4U << 20U;
    limits.max_vertices = 1U << 16U;
    limits.max_faces = 1U << 17U;
    const auto bytes = std::as_bytes(std::span{data, size});
    auto mesh = aam::load_glb(bytes, limits);
    if (mesh) (void)aam::prepare_mesh(*mesh, limits.max_vertices);
    return 0;
}
