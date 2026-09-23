#ifndef FASTFLY_SYCL_BUILD
#define FASTFLY_SYCL_BUILD
#endif
#include "fastfly_sycl.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/filter_selector.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

sycl::device select_device(const char* selector) {
    if (selector && *selector)
        return sycl::ext::oneapi::filter_selector(selector).select_device();
    for (const auto& device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (device.get_info<sycl::info::device::vendor_id>() == 0x8086)
            return device;
    }
    throw std::runtime_error("No Intel SYCL GPU found. Install the Intel GPU driver and oneAPI runtime; use FASTFLY_DEVICE or --device with an explicit SYCL filter (e.g. opencl:cpu) for CPU verification.");
}

// IEEE binary16 storage without requiring device half-arithmetic support.
uint16_t pack_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const int exponent = static_cast<int>((bits >> 23) & 255) - 127;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent < -25) return sign;
    if (exponent < -14) {
        mantissa |= 0x800000;
        const unsigned shift = static_cast<unsigned>(-exponent - 1);
        uint32_t result = mantissa >> shift;
        const uint32_t remainder = mantissa & ((uint32_t{1} << shift) - 1);
        const uint32_t halfway = uint32_t{1} << (shift - 1);
        result += remainder > halfway || (remainder == halfway && (result & 1));
        return static_cast<uint16_t>(sign | result);
    }
    uint32_t result = (static_cast<uint32_t>(exponent + 15) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1fff;
    result += remainder > 0x1000 || (remainder == 0x1000 && (result & 1));
    return static_cast<uint16_t>(sign | result);
}

float unpack_half(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
    const uint32_t exponent = (value >> 10) & 31;
    const uint32_t mantissa = value & 1023;
    if (!exponent) {
        const float magnitude = static_cast<float>(mantissa) * 0x1p-24f;
        return (value & 0x8000) ? -magnitude : magnitude;
    }
    return sycl::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

int8_t quantize(float value, float inverse_scale) {
    if (value == 0.0f || inverse_scale == 0.0f) return 0;
    const float magnitude = std::min(127.0f, std::fabs(value * inverse_scale));
    const float lower = std::floor(magnitude);
    const float fraction = magnitude - lower;
    int rounded = static_cast<int>(lower);
    rounded += fraction > 0.5f || (fraction == 0.5f && (rounded & 1));
    return static_cast<int8_t>(value < 0.0f ? -rounded : rounded);
}

template<class T>
using Atomic = sycl::atomic_ref<T, sycl::memory_order::relaxed,
    sycl::memory_scope::device, sycl::access::address_space::global_space>;

struct Simulation {
    sycl::queue queue;
    std::string device_name;
    std::vector<void*> allocations;
    uint32_t n, words, seed, step = 0, groups = 0, motors = 0;
    size_t local_size = 0;
    unsigned subgroup_size = 0;
    bool int8_weights;
    uint32_t *offsets = nullptr, *targets = nullptr, *spike_bits = nullptr;
    uint32_t *spike_indices = nullptr, *spike_count = nullptr;
    uint16_t* half_weights = nullptr;
    int8_t* byte_weights = nullptr;
    float *scales = nullptr, *voltage = nullptr, *current = nullptr, *stimulus = nullptr;
    float* voltage_sum = nullptr;
    int32_t *group_map = nullptr, *motor_map = nullptr;
    uint64_t *total = nullptr, *group_counts = nullptr, *motor_counts = nullptr;

    Simulation(uint32_t count, uint32_t random_seed, bool quantized, const char* selector)
        : queue(select_device(selector), [](sycl::exception_list errors) {
              if (errors.begin() != errors.end()) std::rethrow_exception(*errors.begin());
          }, sycl::property::queue::in_order{}),
          device_name(queue.get_device().get_info<sycl::info::device::name>()),
          n(count), words(static_cast<uint32_t>((uint64_t(count) + 31) / 32)),
          seed(random_seed), int8_weights(quantized) {
        const auto device = queue.get_device();
        require(device.has(sycl::aspect::usm_device_allocations),
                "Selected device does not support USM device allocations");
        require(device.has(sycl::aspect::atomic64),
                "Selected device does not support 64-bit atomic spike counters");
        const size_t max_group = device.get_info<sycl::info::device::max_work_group_size>();
        const auto max_items = device.get_info<sycl::info::device::max_work_item_sizes<1>>();
        local_size = std::min({size_t{128}, max_group, max_items[0]});
        local_size = local_size / 32 * 32;
        require(local_size >= 32, "Selected device needs a work-group size of at least 32");
        const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        for (unsigned candidate : {16u, 32u, 8u, 64u, 1u}) {
            if (local_size % candidate == 0 &&
                std::find(sizes.begin(), sizes.end(), candidate) != sizes.end()) {
                subgroup_size = candidate;
                break;
            }
        }
        require(subgroup_size != 0, "Selected device has no supported subgroup width (1, 8, 16, 32, 64)");
    }

    ~Simulation() noexcept {
        try { queue.wait_and_throw(); } catch (...) {}
        for (void* ptr : allocations) {
            try { sycl::free(ptr, queue); } catch (...) {}
        }
    }

    template<class T> T* allocate(size_t count) {
        if (!count) return nullptr;
        T* ptr = sycl::malloc_device<T>(count, queue);
        if (!ptr) throw std::bad_alloc();
        try { allocations.push_back(ptr); }
        catch (...) { sycl::free(ptr, queue); throw; }
        return ptr;
    }

    void release(void* ptr) {
        if (!ptr) return;
        sycl::free(ptr, queue);
        allocations.erase(std::find(allocations.begin(), allocations.end(), ptr));
    }

    template<class T> void upload(T* dest, const T* source, size_t count) {
        if (count) queue.memcpy(dest, source, count * sizeof(T)).wait_and_throw();
    }

    void initialize(uint32_t synapses, const uint32_t* host_offsets,
                    const uint32_t* host_targets, const float* host_weights,
                    const float* host_voltage) {
        offsets = allocate<uint32_t>(size_t(n) + 1);
        targets = allocate<uint32_t>(synapses);
        voltage = allocate<float>(n);
        current = allocate<float>(n);
        stimulus = allocate<float>(n);
        voltage_sum = allocate<float>(1);
        spike_bits = allocate<uint32_t>(words);
        spike_indices = allocate<uint32_t>(n);
        spike_count = allocate<uint32_t>(1);
        total = allocate<uint64_t>(1);
        group_map = allocate<int32_t>(n);
        motor_map = allocate<int32_t>(n);
        upload(offsets, host_offsets, size_t(n) + 1);
        upload(targets, host_targets, synapses);
        upload(voltage, host_voltage, n);
        if (int8_weights) {
            byte_weights = allocate<int8_t>(synapses);
            scales = allocate<float>(n);
            std::vector<int8_t> quantized(synapses);
            std::vector<float> row_scales(n);
            for (uint32_t row = 0; row < n; ++row) {
                float maximum = 0.0f;
                for (uint32_t j = host_offsets[row]; j < host_offsets[row + 1]; ++j)
                    maximum = std::max(maximum, std::fabs(host_weights[j]));
                row_scales[row] = maximum / 127.0f;
                const float inverse = row_scales[row] > 0.0f ? 1.0f / row_scales[row] : 0.0f;
                for (uint32_t j = host_offsets[row]; j < host_offsets[row + 1]; ++j)
                    quantized[j] = quantize(host_weights[j], inverse);
            }
            upload(byte_weights, quantized.data(), synapses);
            upload(scales, row_scales.data(), n);
        } else {
            half_weights = allocate<uint16_t>(synapses);
            std::vector<uint16_t> packed(synapses);
            for (uint32_t j = 0; j < synapses; ++j) packed[j] = pack_half(host_weights[j]);
            upload(half_weights, packed.data(), synapses);
        }
        queue.fill(current, 0.0f, n);
        queue.fill(stimulus, 0.0f, n);
        queue.fill(spike_bits, uint32_t{0}, words);
        queue.fill(spike_count, uint32_t{0}, 1);
        queue.fill(total, uint64_t{0}, 1);
        queue.fill(group_map, int32_t{-1}, n);
        queue.fill(motor_map, int32_t{-1}, n);
        queue.wait_and_throw();
    }

    template<unsigned Width> void propagate() {
        auto idx = spike_indices;
        auto count = spike_count;
        auto off = offsets;
        auto tgt = targets;
        auto half = half_weights;
        auto bytes = byte_weights;
        auto scale = scales;
        auto input = current;
        const bool quantized = int8_weights;
        const size_t workgroups = std::min(size_t{2048},
            std::max(size_t{1}, (size_t(n) + local_size / Width - 1) / (local_size / Width)));
        const size_t global = workgroups * local_size;
        queue.parallel_for(sycl::nd_range<1>(global, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Width)]] {
                const auto subgroup = item.get_sub_group();
                const size_t lane = subgroup.get_local_linear_id();
                const size_t group = item.get_group_linear_id() * subgroup.get_group_linear_range()
                                   + subgroup.get_group_linear_id();
                const size_t stride = item.get_group_range(0) * subgroup.get_group_linear_range();
                for (size_t spike = group; spike < *count; spike += stride) {
                    const uint32_t neuron = idx[spike];
                    const size_t end = off[neuron + 1];
                    const float multiplier = quantized ? scale[neuron] : 1.0f;
                    for (size_t j = size_t(off[neuron]) + lane; j < end; j += Width) {
                        const float weight = quantized ? static_cast<float>(bytes[j]) * multiplier
                                                       : unpack_half(half[j]);
                        Atomic<float>(input[tgt[j]]).fetch_add(weight);
                    }
                }
            });
    }

    void run(uint32_t steps, float decay, float threshold, float reset, float noise) {
        queue.fill(total, uint64_t{0}, 1);
        if (groups) queue.fill(group_counts, uint64_t{0}, groups);
        if (motors) queue.fill(motor_counts, uint64_t{0}, motors);
        const auto v = voltage;
        const auto input = current;
        const auto stim = stimulus;
        const auto bits = spike_bits;
        const auto idx = spike_indices;
        const auto count = spike_count;
        const auto batch_total = total;
        const auto gm = group_map;
        const auto mm = motor_map;
        const auto gc = group_counts;
        const auto mc = motor_counts;
        const uint32_t neuron_count = n;
        const uint32_t word_count = words;
        const uint32_t random_seed = seed;
        const size_t local = local_size;
        const size_t global = (size_t(n) + local - 1) / local * local;
        for (uint32_t iteration = 0; iteration < steps; ++iteration) {
            const uint32_t current_step = step++;
            queue.fill(count, uint32_t{0}, 1);
            queue.submit([&](sycl::handler& handler) {
                sycl::local_accessor<uint32_t, 1> flags(sycl::range<1>(local), handler);
                handler.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_linear_id();
                    const size_t lane = item.get_local_linear_id();
                    bool spiked = false;
                    if (i < neuron_count) {
                        uint32_t h = static_cast<uint32_t>(i) ^ (current_step * 2654435761u)
                                   ^ (random_seed * 1664525u);
                        h ^= h >> 16;
                        h *= 0x45d9f3bu;
                        h ^= h >> 16;
                        const float random_input = noise * (static_cast<float>(h & 0xffff) / 32768.0f - 1.0f);
                        const float next = v[i] * decay + ((input[i] + stim[i]) + random_input);
                        spiked = next >= threshold;
                        v[i] = spiked ? reset : next;
                        input[i] = 0.0f;
                    }
                    flags[lane] = spiked ? (uint32_t{1} << (lane % 32)) : 0;
                    sycl::group_barrier(item.get_group());
                    if (lane % 32 == 0 && i / 32 < word_count) {
                        uint32_t word = 0;
                        for (size_t j = 0; j < 32; ++j) word |= flags[lane + j];
                        bits[i / 32] = word;
                    }
                });
            });
            queue.parallel_for(sycl::range<1>(word_count), [=](sycl::id<1> item) {
                uint32_t word = bits[item[0]];
                if (!word) return;
                const uint32_t fired = sycl::popcount(word);
                uint32_t position = Atomic<uint32_t>(*count).fetch_add(fired);
                Atomic<uint64_t>(*batch_total).fetch_add(fired);
                while (word) {
                    const uint32_t bit = 31u - sycl::clz(word & (~word + 1u));
                    const uint32_t neuron = static_cast<uint32_t>(item[0] * 32 + bit);
                    idx[position++] = neuron;
                    const int32_t group = gm[neuron];
                    const int32_t motor = mm[neuron];
                    if (group >= 0) Atomic<uint64_t>(gc[group]).fetch_add(1);
                    if (motor >= 0) Atomic<uint64_t>(mc[motor]).fetch_add(1);
                    word &= word - 1;
                }
            });
            switch (subgroup_size) {
                case 1: propagate<1>(); break;
                case 8: propagate<8>(); break;
                case 16: propagate<16>(); break;
                case 32: propagate<32>(); break;
                case 64: propagate<64>(); break;
            }
        }
        queue.wait_and_throw();
    }
};

Simulation& get(void* handle) {
    require(handle != nullptr, "Simulation handle is null");
    return *static_cast<Simulation*>(handle);
}

template<class Function> int status(Function&& function) noexcept {
    last_error.clear();
    try { function(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); }
    catch (...) { last_error = "Unknown native SYCL error"; }
    return -1;
}
} // namespace

extern "C" {
const char* ff_last_error(void) { return last_error.c_str(); }

void* ff_create(uint32_t n, uint32_t s, const uint32_t* offsets,
                const uint32_t* targets, const float* weights, const float* voltage,
                uint32_t seed, int int8_weights, const char* device_selector) {
    Simulation* result = nullptr;
    status([&] {
        require(n > 0, "Neuron count must be positive");
        require(offsets && voltage && (!s || (targets && weights)), "Missing CSR or voltage buffer");
        require(int8_weights == 0 || int8_weights == 1, "Weight mode must be 0 (FP16) or 1 (INT8)");
        require(offsets[0] == 0 && offsets[n] == s, "CSR offsets must start at zero and end at the synapse count");
        for (uint32_t i = 0; i < n; ++i) {
            require(offsets[i] <= offsets[i + 1] && offsets[i + 1] <= s, "CSR offsets must be monotonic and in bounds");
            require(std::isfinite(voltage[i]), "Initial voltage must be finite");
        }
        for (uint32_t j = 0; j < s; ++j) {
            require(targets[j] < n, "CSR target index out of bounds");
            require(std::isfinite(weights[j]), "Synaptic weights must be finite");
            require(int8_weights || std::fabs(weights[j]) <= 65504.0f, "Synaptic weight exceeds finite FP16 range");
        }
        auto sim = std::make_unique<Simulation>(n, seed, int8_weights != 0, device_selector);
        sim->initialize(s, offsets, targets, weights, voltage);
        result = sim.release();
    });
    return result;
}

void ff_destroy(void* handle) {
    status([&] { delete static_cast<Simulation*>(handle); });
}

const char* ff_device_name(void* handle) {
    const char* name = nullptr;
    status([&] { name = get(handle).device_name.c_str(); });
    return name;
}

int ff_set_groups(void* handle, const int32_t* group_map, uint32_t groups,
                  const int32_t* motor_map, uint32_t motors) {
    return status([&] {
        auto& sim = get(handle);
        require(groups <= uint32_t(INT32_MAX) && motors <= uint32_t(INT32_MAX), "Too many metric categories");
        require((!groups || group_map) && (!motors || motor_map), "Missing metric category map");
        for (uint32_t i = 0; i < sim.n; ++i) {
            if (group_map) require(group_map[i] >= -1 && (group_map[i] < 0 || uint32_t(group_map[i]) < groups), "Group index out of bounds");
            if (motor_map) require(motor_map[i] >= -1 && (motor_map[i] < 0 || uint32_t(motor_map[i]) < motors), "Motor index out of bounds");
        }
        sim.queue.wait_and_throw();
        uint64_t* new_groups = sim.allocate<uint64_t>(groups);
        uint64_t* new_motors = nullptr;
        try { new_motors = sim.allocate<uint64_t>(motors); }
        catch (...) { sim.release(new_groups); throw; }
        sim.release(sim.group_counts);
        sim.release(sim.motor_counts);
        sim.group_counts = new_groups;
        sim.motor_counts = new_motors;
        sim.groups = groups;
        sim.motors = motors;
        if (group_map) sim.upload(sim.group_map, group_map, sim.n);
        else sim.queue.fill(sim.group_map, int32_t{-1}, sim.n);
        if (motor_map) sim.upload(sim.motor_map, motor_map, sim.n);
        else sim.queue.fill(sim.motor_map, int32_t{-1}, sim.n);
        if (groups) sim.queue.fill(sim.group_counts, uint64_t{0}, groups);
        if (motors) sim.queue.fill(sim.motor_counts, uint64_t{0}, motors);
        sim.queue.wait_and_throw();
    });
}

int ff_set_stimulus(void* handle, const uint32_t* unique_indices,
                    uint32_t count, float amplitude) {
    return status([&] {
        auto& sim = get(handle);
        require(std::isfinite(amplitude), "Stimulus amplitude must be finite");
        require(count <= sim.n && (!count || unique_indices), "Invalid stimulus index buffer");
        std::vector<uint32_t> sorted;
        if (count) sorted.assign(unique_indices, unique_indices + count);
        std::sort(sorted.begin(), sorted.end());
        require(sorted.empty() || sorted.back() < sim.n, "Stimulus index out of bounds");
        require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "Stimulus indices must be unique");
        std::vector<float> values(sim.n, 0.0f);
        for (uint32_t index : sorted) values[index] = amplitude;
        sim.upload(sim.stimulus, values.data(), sim.n);
    });
}

int ff_step(void* handle, uint32_t steps, float decay, float threshold, float reset, float noise) {
    return status([&] {
        require(std::isfinite(decay) && std::isfinite(threshold) && std::isfinite(reset) && std::isfinite(noise),
                "Neuron parameters must be finite");
        get(handle).run(steps, decay, threshold, reset, noise);
    });
}

int ff_read_metrics(void* handle, uint64_t* total, uint64_t* groups,
                    uint64_t* motors, float* mean_voltage) {
    return status([&] {
        auto& sim = get(handle);
        require(total && mean_voltage && (!sim.groups || groups) && (!sim.motors || motors), "Missing metrics output buffer");
        const auto voltage = sim.voltage;
        sim.queue.submit([&](sycl::handler& handler) {
            auto sum = sycl::reduction(sim.voltage_sum, sycl::plus<float>(),
                sycl::property::reduction::initialize_to_identity{});
            handler.parallel_for(sycl::range<1>(sim.n), sum,
                [=](sycl::id<1> i, auto& accumulator) { accumulator.combine(voltage[i]); });
        });
        sim.queue.memcpy(total, sim.total, sizeof(uint64_t)).wait_and_throw();
        if (sim.groups) sim.queue.memcpy(groups, sim.group_counts, sizeof(uint64_t) * sim.groups).wait_and_throw();
        if (sim.motors) sim.queue.memcpy(motors, sim.motor_counts, sizeof(uint64_t) * sim.motors).wait_and_throw();
        sim.queue.memcpy(mean_voltage, sim.voltage_sum, sizeof(float)).wait_and_throw();
        *mean_voltage /= sim.n;
    });
}

int ff_read_spikes(void* handle, uint32_t* indices, uint32_t* count) {
    return status([&] {
        auto& sim = get(handle);
        require(indices && count, "Missing spike output buffer");
        sim.queue.memcpy(count, sim.spike_count, sizeof(uint32_t)).wait_and_throw();
        if (*count) sim.queue.memcpy(indices, sim.spike_indices, sizeof(uint32_t) * *count).wait_and_throw();
    });
}

int ff_read_state(void* handle, float* voltage, float* current) {
    return status([&] {
        auto& sim = get(handle);
        require(voltage && current, "Missing state output buffer");
        sim.queue.memcpy(voltage, sim.voltage, sizeof(float) * sim.n).wait_and_throw();
        sim.queue.memcpy(current, sim.current, sizeof(float) * sim.n).wait_and_throw();
    });
}
} // extern "C"
