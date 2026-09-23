#include "fastfly_sycl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Config {
    uint32_t neurons = 139255, synapses = 54500000, timesteps = 10000;
    uint32_t warmup = 500, seed = 42;
    bool verbose = false;
    std::string data, device;
};

struct Connectome {
    uint32_t n = 0, s = 0;
    std::vector<uint32_t> offsets, targets;
    std::vector<float> weights;
};

uint32_t number(const char* text, const std::string& option) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(option + " requires a nonnegative integer");
    const unsigned long long parsed = std::stoull(value);
    if (parsed > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(option + " exceeds the uint32 range");
    return static_cast<uint32_t>(parsed);
}

void check(int result) {
    if (result != 0) throw std::runtime_error(ff_last_error());
}

template<class T> void read_array(std::ifstream& file, T* values, size_t count) {
    if (count && !file.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(sizeof(T) * count)))
        throw std::runtime_error("Truncated FlyWire binary file");
}

Connectome load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open '" + filename + "'; run python download_connectome.py first");
    const auto file_size = file.tellg();
    file.seekg(0);
    uint32_t header[4];
    read_array(file, header, 4);
    if (header[0] != 0x464c5957u) throw std::runtime_error("Invalid FlyWire file magic");
    if (header[1] != 1) throw std::runtime_error("Unsupported FlyWire binary version (expected 1)");
    Connectome graph;
    graph.n = header[2];
    graph.s = header[3];
    if (!graph.n) throw std::runtime_error("Connectome must contain at least one neuron");
    const uint64_t expected = 16 + (uint64_t(graph.n) + 1) * 4 + uint64_t(graph.s) * 8;
    if (file_size < 0 || static_cast<uint64_t>(file_size) != expected)
        throw std::runtime_error("FlyWire binary size does not match its header");
    graph.offsets.resize(size_t(graph.n) + 1);
    graph.targets.resize(graph.s);
    graph.weights.resize(graph.s);
    read_array(file, graph.offsets.data(), graph.offsets.size());
    read_array(file, graph.targets.data(), graph.targets.size());
    read_array(file, graph.weights.data(), graph.weights.size());
    return graph;
}

Connectome generate(uint32_t n, uint32_t s, uint32_t seed) {
    if (n == 0 || (n == 1 && s != 0))
        throw std::invalid_argument("Synthetic connectivity requires positive neurons and no synapses for a single neuron");
    Connectome graph;
    graph.n = n;
    graph.s = s;
    graph.offsets.resize(size_t(n) + 1, 0);
    graph.targets.resize(s);
    graph.weights.resize(s);
    if (!s) return graph;
    std::mt19937 rng(seed);
    const double mean = static_cast<double>(s) / n;
    std::lognormal_distribution<double> degree_distribution(std::log(mean) - 0.5 * std::log(5.0), std::sqrt(std::log(5.0)));
    std::vector<uint32_t> degrees(n);
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        degrees[i] = static_cast<uint32_t>(std::max(1.0, std::min(double(n - 1), degree_distribution(rng))));
        total += degrees[i];
    }
    const double scale = double(s) / total;
    total = 0;
    for (auto& degree : degrees) {
        // Sparse smoke graphs may have empty rows; real defaults keep a
        // minimum degree of one, matching the original Dale-law generator.
        degree = static_cast<uint32_t>(std::min(double(s),
            std::max(s >= n ? 1.0 : 0.0, std::round(degree * scale))));
        total += degree;
    }
    // Targets are sampled with replacement, so parallel synapses are valid.
    while (total != s) {
        auto& degree = degrees[rng() % n];
        if (total < s) { ++degree; ++total; }
        else if (degree > (s >= n ? 1u : 0u)) { --degree; --total; }
    }
    for (uint32_t i = 0; i < n; ++i) graph.offsets[i + 1] = graph.offsets[i] + degrees[i];
    std::uniform_int_distribution<uint32_t> target_distribution(0, n - 2);
    std::normal_distribution<float> weight_distribution(0.0f, 0.03f);
    std::bernoulli_distribution excitatory_distribution(0.7);
    std::vector<std::pair<uint32_t, float>> row;
    for (uint32_t i = 0; i < n; ++i) {
        const bool excitatory = excitatory_distribution(rng);
        row.resize(degrees[i]);
        for (auto& synapse : row) {
            const uint32_t target = target_distribution(rng);
            synapse.first = target >= i ? target + 1 : target;
            const float weight = std::fabs(weight_distribution(rng)) + 0.005f;
            synapse.second = excitatory ? weight : -weight;
        }
        std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t j = 0; j < row.size(); ++j) {
            const size_t index = size_t(graph.offsets[i]) + j;
            graph.targets[index] = row[j].first;
            graph.weights[index] = row[j].second;
        }
    }
    return graph;
}

void help() {
    std::puts("FlyWire Connectome Intel oneAPI SYCL Simulator\n\n"
              "Usage: flywire_sim [options]\n\n"
              "  --data FILE    Load real FlyWire v1 binary connectome\n"
              "  --timesteps N  Benchmark timesteps (default: 10000)\n"
              "  --warmup N     Warmup timesteps (default: 500)\n"
              "  --seed N       Random seed (default: 42)\n"
              "  --neurons N    Synthetic neuron count (default: 139255)\n"
              "  --synapses N   Synthetic synapse count (default: 54500000)\n"
              "  --device FILTER  Explicit SYCL filter, e.g. level_zero:gpu:0\n"
              "                   or opencl:cpu for verification\n"
              "  --verbose      Print every timestep (synchronizes each step)\n"
              "  --help         Show this help\n\n"
              "Default device: Intel GPU only. FASTFLY_DEVICE also sets the filter.\n"
              "Native weights use FP16 storage. Wall timings include submission and synchronization.");
}
} // namespace

int main(int argc, char** argv) {
    try {
        Config config;
        if (const char* selector = std::getenv("FASTFLY_DEVICE")) config.device = selector;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--help") { help(); return 0; }
            if (argument == "--verbose") { config.verbose = true; continue; }
            if (argument != "--data" && argument != "--device" && argument != "--timesteps" &&
                argument != "--warmup" && argument != "--seed" && argument != "--neurons" && argument != "--synapses")
                throw std::invalid_argument("Unknown option: " + argument);
            if (++i == argc) throw std::invalid_argument("Missing value for " + argument);
            if (argument == "--data") config.data = argv[i];
            else if (argument == "--device") config.device = argv[i];
            else if (argument == "--timesteps") config.timesteps = number(argv[i], argument);
            else if (argument == "--warmup") config.warmup = number(argv[i], argument);
            else if (argument == "--seed") config.seed = number(argv[i], argument);
            else if (argument == "--neurons") config.neurons = number(argv[i], argument);
            else if (argument == "--synapses") config.synapses = number(argv[i], argument);
        }
        if (!config.timesteps) throw std::invalid_argument("--timesteps must be positive");
        std::puts("\nFlyWire Connectome Intel oneAPI SYCL Simulator\n");
        std::printf("DATA SOURCE: %s\n", config.data.empty() ? "Synthetic (matching FlyWire statistics)" : config.data.c_str());
        const auto load_start = std::chrono::steady_clock::now();
        auto graph = config.data.empty() ? generate(config.neurons, config.synapses, config.seed) : load(config.data);
        std::printf("Connectome: %u neurons, %u synapses\n", graph.n, graph.s);
        std::printf("Data loading/generation: %.3f seconds\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count());
        std::vector<float> initial_voltage(graph.n, 0.0f);
        using Handle = std::unique_ptr<void, decltype(&ff_destroy)>;
        Handle simulation(ff_create(graph.n, graph.s, graph.offsets.data(), graph.targets.data(),
                                    graph.weights.data(), initial_voltage.data(), config.seed, 0,
                                    config.device.c_str()), &ff_destroy);
        if (!simulation) throw std::runtime_error(ff_last_error());
        const uint32_t neurons = graph.n;
        const uint32_t synapses = graph.s;
        graph = Connectome{};
        std::vector<float>().swap(initial_voltage);
        const char* device_name = ff_device_name(simulation.get());
        if (!device_name) throw std::runtime_error(ff_last_error());
        std::printf("Device: %s\nWeights: FP16\n", device_name);
        std::printf("Running simulation: %u warmup + %u benchmark timesteps\n", config.warmup, config.timesteps);
        const uint32_t batch_size = config.verbose ? 1 : 1000;
        uint64_t benchmark_spikes = 0;
        double benchmark_seconds = 0.0;
        float mean_voltage = 0.0f;
        for (int phase = 0; phase < 2; ++phase) {
            const uint32_t limit = phase ? config.timesteps : config.warmup;
            for (uint32_t step = 0; step < limit;) {
                const uint32_t batch = std::min(batch_size, limit - step);
                const auto start = std::chrono::steady_clock::now();
                check(ff_step(simulation.get(), batch, 0.9f, 1.0f, 0.0f, 0.15f));
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                uint64_t spikes = 0;
                check(ff_read_metrics(simulation.get(), &spikes, nullptr, nullptr, &mean_voltage));
                if (phase) { benchmark_spikes += spikes; benchmark_seconds += elapsed; }
                step += batch;
                std::printf("%s %u/%u  wall %.2f us/step  spikes/step %.2f\n", phase ? "BENCH" : "WARM",
                            step, limit, elapsed * 1e6 / batch, double(spikes) / batch);
            }
        }
        const double microseconds = benchmark_seconds * 1e6 / config.timesteps;
        const double average_spikes = double(benchmark_spikes) / config.timesteps;
        const double speedup = config.timesteps * 0.001 / benchmark_seconds;
        std::printf("\nBENCHMARK RESULTS (averaged over %u timesteps)\n", config.timesteps);
        std::printf("  Total wall time:       %.6f seconds\n", benchmark_seconds);
        std::printf("  Wall time per step:    %.3f us (submission + execution + synchronization)\n", microseconds);
        std::printf("  Avg spikes/step:       %.2f (%.4f%% firing rate)\n", average_spikes, 100.0 * average_spikes / neurons);
        std::printf("  Mean final voltage:    %.6f\n", mean_voltage);
        std::printf("  Biological time per wall-second: %.1f ms\n", 1e6 / microseconds);
        std::printf("  Speed vs real-time:    %.3fx\n", speedup);
        std::printf("  Avg degree:            %.2f synapses/neuron\n", double(synapses) / neurons);
        std::puts("  Wall timings exclude metric readback and terminal output; no kernel-only timing claimed.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
