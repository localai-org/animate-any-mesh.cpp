#include "aam/aam.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    const char * model_path = std::getenv("AAM_TEST_GGUF");
    if (model_path == nullptr) {
        std::cout << "SKIP: set AAM_TEST_GGUF\n";
        return 77;
    }
    aam::mesh input;
    input.vertices = {
        {-1.0F, -1.0F, -1.0F}, {1.0F, -1.0F, 1.0F},
        {-1.0F, 1.0F, 1.0F}, {1.0F, 1.0F, -1.0F}};
    input.faces = {{0U, 1U, 2U}, {0U, 3U, 1U}, {0U, 2U, 3U}, {1U, 3U, 2U}};
    aam::runtime_options runtime;
    runtime.device = aam::device_kind::cpu;
    runtime.threads = 8U;
    auto model = aam::model::load(model_path, runtime);
    if (!model) { std::cerr << model.error().message << '\n'; return 1; }
    aam::generation_options options;
    options.seed = 1234U;
    options.trajectories = 4U;
    options.steps = 2U;
    auto generated = model->generate(input, "The object is flying", options);
    if (!generated) { std::cerr << generated.error().message << '\n'; return 1; }
    if (generated->frames != 16U || generated->vertices_per_frame != input.vertices.size() ||
        generated->positions.size() != 16U * input.vertices.size()) return 1;
    for (const auto & position : generated->positions) {
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) return 1;
    }
    auto prepared = aam::prepare_mesh(input, options.max_vertices);
    if (!prepared) { std::cerr << prepared.error().message << '\n'; return 1; }
    auto reused = model->generate(*prepared, "The object is flying", options);
    if (!reused) { std::cerr << reused.error().message << '\n'; return 1; }
    if (reused->frames != generated->frames ||
        reused->vertices_per_frame != generated->vertices_per_frame ||
        reused->positions != generated->positions) return 1;
    prepared->original_to_merged.front() =
        static_cast<std::uint32_t>(prepared->value.vertices.size());
    if (model->generate(*prepared, "The object is flying", options)) return 1;
    std::cout << "generation smoke test passed\n";
    return 0;
}
