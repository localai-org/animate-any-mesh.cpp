#include "aam/aam.hpp"

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <string>
#include <string_view>

namespace {

void usage(std::ostream & out) {
    out << "Usage:\n"
        << "  aam-cli backends [--backend-dir DIR]\n"
        << "  aam-cli animate --model MODEL.gguf --input MESH.glb --output ANIMATED.glb\n"
        << "                  --prompt TEXT [--device auto|cpu|vulkan] [--seed N]\n"
        << "                  [--trajectories N] [--steps N] [--guidance X]\n"
        << "                  [--threads N] [--fps X] [--max-vertices N]\n"
        << "                  [--warmups N] [--benchmark-runs N]\n"
        << "                  [--fast-fps]\n"
        << "                  [--backend-dir DIR]\n";
}

struct arguments {
    std::string model;
    std::string input;
    std::string output;
    std::string prompt;
    std::string backend_directory;
    aam::device_kind device = aam::device_kind::automatic;
    std::uint64_t seed = 666U;
    std::size_t trajectories = 0U;
    std::size_t steps = 64U;
    float guidance = 3.0F;
    float fps = 24.0F;
    std::uint32_t threads = 0U;
    std::size_t max_vertices = aam::default_max_vertices;
    std::size_t warmups = 0U;
    std::size_t benchmark_runs = 1U;
    bool strict_fps_parity = true;
};

template<class T>
bool parse_number(std::string_view text, T & value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

std::optional<arguments> parse_animate(int argc, char ** argv) {
    arguments result;
    for (int index = 2; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--fast-fps") {
            result.strict_fps_parity = false;
            continue;
        }
        if (index + 1 >= argc) {
            std::cerr << "missing value for " << option << '\n';
            return std::nullopt;
        }
        const std::string_view value{argv[++index]};
        if (option == "--model") result.model = value;
        else if (option == "--input") result.input = value;
        else if (option == "--output") result.output = value;
        else if (option == "--prompt") result.prompt = value;
        else if (option == "--backend-dir") result.backend_directory = value;
        else if (option == "--device") {
            if (value == "auto") result.device = aam::device_kind::automatic;
            else if (value == "cpu") result.device = aam::device_kind::cpu;
            else if (value == "vulkan") result.device = aam::device_kind::vulkan;
            else { std::cerr << "invalid device: " << value << '\n'; return std::nullopt; }
        } else if (option == "--seed") {
            if (!parse_number(value, result.seed)) return std::nullopt;
        } else if (option == "--trajectories") {
            if (!parse_number(value, result.trajectories)) return std::nullopt;
        } else if (option == "--steps") {
            if (!parse_number(value, result.steps)) return std::nullopt;
        } else if (option == "--guidance") {
            if (!parse_number(value, result.guidance)) return std::nullopt;
        } else if (option == "--threads") {
            if (!parse_number(value, result.threads)) return std::nullopt;
        } else if (option == "--fps") {
            if (!parse_number(value, result.fps)) return std::nullopt;
        } else if (option == "--max-vertices") {
            if (!parse_number(value, result.max_vertices)) return std::nullopt;
        } else if (option == "--warmups") {
            if (!parse_number(value, result.warmups)) return std::nullopt;
        } else if (option == "--benchmark-runs") {
            if (!parse_number(value, result.benchmark_runs)) return std::nullopt;
        } else {
            std::cerr << "unknown option: " << option << '\n';
            return std::nullopt;
        }
    }
    if (result.model.empty() || result.input.empty() || result.output.empty() ||
        result.prompt.empty()) {
        std::cerr << "--model, --input, --output, and --prompt are required\n";
        return std::nullopt;
    }
    constexpr std::size_t maximum_benchmark_iterations = 1000U;
    if (result.benchmark_runs == 0U || result.warmups > maximum_benchmark_iterations ||
        result.benchmark_runs > maximum_benchmark_iterations) {
        std::cerr << "--benchmark-runs must be in [1, 1000] and --warmups in [0, 1000]\n";
        return std::nullopt;
    }
    return result;
}

int report(const aam::error & error) {
    std::cerr << "error: " << error.message << '\n';
    return EXIT_FAILURE;
}

} // namespace

int run(int argc, char ** argv) {
    if (argc >= 2 && std::string_view{argv[1]} == "backends") {
        std::filesystem::path directory;
        if (argc == 4 && std::string_view{argv[2]} == "--backend-dir") directory = argv[3];
        else if (argc != 2) { usage(std::cerr); return EXIT_FAILURE; }
        for (const auto & backend : aam::available_backends(directory)) std::cout << backend << '\n';
        return EXIT_SUCCESS;
    }
    if (argc < 2 || std::string_view{argv[1]} != "animate") {
        usage(std::cerr);
        return EXIT_FAILURE;
    }
    auto options = parse_animate(argc, argv);
    if (!options) { usage(std::cerr); return EXIT_FAILURE; }

    auto mesh = aam::load_glb_file(options->input);
    if (!mesh) return report(mesh.error());
    auto topology = aam::simplify_mesh(*mesh, options->max_vertices);
    if (!topology) return report(topology.error());
    if (topology->vertices.size() != mesh->vertices.size() ||
        topology->faces.size() != mesh->faces.size()) {
        std::cerr << "simplified mesh: " << mesh->vertices.size() << " -> "
                  << topology->vertices.size() << " vertices, "
                  << mesh->faces.size() << " -> " << topology->faces.size()
                  << " faces\n";
    }
    auto prepared = aam::prepare_mesh(*topology, options->max_vertices);
    if (!prepared) return report(prepared.error());
    aam::runtime_options runtime;
    runtime.device = options->device;
    runtime.threads = options->threads;
    runtime.backend_directory = options->backend_directory;
    runtime.strict_fps_parity = options->strict_fps_parity;
    auto model = aam::model::load(options->model, runtime);
    if (!model) return report(model.error());
    std::cerr << "backend: " << model->backend_name() << '\n';
    aam::generation_options generation;
    generation.seed = options->seed;
    generation.trajectories = options->trajectories;
    generation.steps = options->steps;
    generation.guidance_scale = options->guidance;
    generation.max_vertices = options->max_vertices;
    std::optional<aam::animation> animation;
    const bool benchmark_enabled = options->warmups != 0U || options->benchmark_runs != 1U;
    for (std::size_t run = 0; run < options->warmups + options->benchmark_runs; ++run) {
        const bool measured = run >= options->warmups;
        const auto started = std::chrono::steady_clock::now();
        auto generated = model->generate(*prepared, options->prompt, generation);
        if (!generated) return report(generated.error());
        const auto stopped = std::chrono::steady_clock::now();
        if (benchmark_enabled) {
            if (measured) {
                const std::chrono::duration<double> elapsed = stopped - started;
                std::cerr << "run " << (run - options->warmups + 1U)
                          << " seconds " << elapsed.count() << '\n';
            } else {
                std::cerr << "warmup " << (run + 1U) << " complete\n";
            }
        }
        if (measured) animation = std::move(*generated);
    }
    if (!animation) {
        std::cerr << "error: benchmark produced no animation\n";
        return EXIT_FAILURE;
    }
    auto saved = aam::save_animation_glb_file(
        options->output, *topology, *animation, options->fps);
    if (!saved) return report(saved.error());
    std::cout << options->output << '\n';
    return EXIT_SUCCESS;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::bad_alloc &) {
        std::cerr << "error: not enough memory to complete the request\n";
    } catch (const std::exception & exception) {
        std::cerr << "error: unexpected failure: " << exception.what() << '\n';
    }
    return EXIT_FAILURE;
}
