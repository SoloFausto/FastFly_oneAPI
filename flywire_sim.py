"""FlyWire connectome benchmark using the native Intel oneAPI SYCL backend.

Python uses per-row INT8 synapse weights. Library loading and device selection
are deferred until a simulation is created, so imports and --help need only NumPy.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

from sycl_backend import NativeSimulation


def load_connectome_binary(filename):
    print(f"Loading connectome from '{filename}'...")
    with open(filename, "rb") as source:
        header = source.read(16)
        if len(header) != 16:
            raise ValueError("Truncated connectome header")
        magic, version, n_neurons, n_synapses = struct.unpack("<IIII", header)
        if magic != 0x464C5957:
            raise ValueError(f"Invalid file magic 0x{magic:X}")
        if version != 1:
            raise ValueError(f"Unsupported connectome version {version}")
        if n_neurons == 0:
            raise ValueError("Connectome must contain at least one neuron")
        arrays = []
        for name, count, dtype in (("offsets", n_neurons + 1, "<u4"),
                                   ("targets", n_synapses, "<u4"),
                                   ("weights", n_synapses, "<f4")):
            payload = source.read(count * 4)
            if len(payload) != count * 4:
                raise ValueError(f"Truncated connectome {name}")
            arrays.append(np.frombuffer(payload, dtype=dtype).copy())
        offsets, targets, weights = arrays
    if (offsets[0] != 0 or offsets[-1] != n_synapses or
            np.any(offsets[1:] < offsets[:-1]) or np.any(targets >= n_neurons)):
        raise ValueError("Invalid connectome CSR indices")
    degrees = np.diff(offsets)
    sample = weights[:min(n_synapses, 1000000)]
    exc_percent = 100 * np.count_nonzero(sample > 0) / sample.size if sample.size else 0.0
    print(f"  Neurons:  {n_neurons}\n  Synapses: {n_synapses}")
    print(f"  Out-degree: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.1f}")
    print(f"  Excitatory (sample): {exc_percent:.1f}%\n  Loaded successfully.\n")
    return n_neurons, n_synapses, offsets, targets, weights


def generate_synthetic(n_neurons=139255, n_synapses=54500000, seed=42):
    if not 1 <= n_neurons <= 0xFFFFFFFF or not 0 <= n_synapses <= 0xFFFFFFFF:
        raise ValueError("Synthetic sizes require 1..4294967295 neurons and 0..4294967295 synapses")
    print(f"Generating synthetic connectome ({n_neurons} neurons, {n_synapses} synapses)...")
    rng = np.random.default_rng(seed)
    # Apportion the exact edge budget, allowing empty rows in small graphs.
    mass = rng.lognormal(-0.5 * np.log(5.0), np.sqrt(np.log(5.0)), n_neurons)
    expected = mass * (n_synapses / mass.sum())
    degrees = np.floor(expected).astype(np.int64)
    remainder = n_synapses - int(degrees.sum())
    if remainder:
        extra = np.argpartition(expected - degrees, n_neurons - remainder)[-remainder:]
        degrees[extra] += 1
    offsets = np.zeros(n_neurons + 1, dtype=np.uint32)
    np.cumsum(degrees, out=offsets[1:])
    targets = rng.integers(0, n_neurons, n_synapses, dtype=np.uint32)
    excitatory = rng.random(n_neurons) < 0.7
    weights = np.abs(rng.normal(0, 0.03, n_synapses)).astype(np.float32) + 0.005
    for i in range(n_neurons):
        if not excitatory[i]:
            weights[offsets[i]:offsets[i + 1]] *= -1
    print(f"  Generated {n_synapses} synapses")
    print(f"  Degree: min={degrees.min()}, max={degrees.max()}, mean={degrees.mean():.1f}\n")
    return n_neurons, n_synapses, offsets, targets, weights


def quantize_weights_int8(weights, offsets, n_neurons):
    """Return INT8 weights and per-row FP32 scales, including empty CSR rows.

    Round to nearest, ties to even, matching the native backend's INT8 mode.
    """
    weights = np.asarray(weights, dtype=np.float32)
    offsets = np.asarray(offsets)
    if (weights.ndim != 1 or offsets.shape != (n_neurons + 1,) or
            offsets.dtype.kind not in "iu" or offsets[0] != 0 or offsets[-1] != weights.size or
            np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("Invalid CSR offsets for quantization")
    degrees = np.diff(offsets).astype(np.intp)
    nonempty = degrees > 0
    max_abs = np.zeros(n_neurons, dtype=np.float32)
    if weights.size:
        starts = offsets[:-1][nonempty].astype(np.intp)
        max_abs[nonempty] = np.maximum.reduceat(np.abs(weights), starts)
    scales = (max_abs / np.float32(127.0)).astype(np.float32)
    inv_scales = np.zeros_like(scales)
    np.divide(np.float32(1.0), scales, out=inv_scales, where=scales > 0)
    per_syn_inv_scale = np.repeat(inv_scales, degrees)
    quantized = np.clip(np.round(weights * per_syn_inv_scale), -127, 127).astype(np.int8)
    return quantized, scales


def print_gpu_info(simulation=None):
    if simulation is None:
        with NativeSimulation([0, 0], [], [], [0.0]) as probe:
            print_gpu_info(probe)
        return
    print("=" * 60)
    print(f"SYCL device: {simulation.device_name}")
    print("Synapse storage: INT8 with per-neuron FP32 scales")
    print("=" * 60 + "\n")


def run_simulation(n_neurons, n_synapses, offsets, targets, weights,
                   num_timesteps=10000, warmup_steps=500, seed=42, verbose=False):
    if num_timesteps <= 0 or warmup_steps < 0:
        raise ValueError("timesteps must be positive and warmup must be nonnegative")
    if len(offsets) != n_neurons + 1 or len(targets) != n_synapses or len(weights) != n_synapses:
        raise ValueError("Connectome sizes do not match its CSR arrays")
    voltage = np.random.default_rng(seed).uniform(0.0, 0.9, n_neurons).astype(np.float32)
    started = time.perf_counter()
    with NativeSimulation(offsets, targets, weights, voltage, seed=seed, int8_weights=True) as simulation:
        print_gpu_info(simulation)
        print(f"  Upload time: {time.perf_counter() - started:.2f}s\n")
        print(f"Running: {warmup_steps} warmup + {num_timesteps} benchmark timesteps")
        if warmup_steps:
            simulation.step(warmup_steps)
        total_spikes = 0
        elapsed = 0.0
        batch_size = 1 if verbose else 200
        for start in range(0, num_timesteps, batch_size):
            steps = min(batch_size, num_timesteps - start)
            started = time.perf_counter()
            simulation.step(steps)
            metrics = simulation.read_metrics()
            batch_elapsed = time.perf_counter() - started
            elapsed += batch_elapsed
            total_spikes += metrics["total_spikes"]
            if verbose or start % 1000 == 0:
                rate = 100.0 * metrics["total_spikes"] / (steps * n_neurons)
                print(f"BENCH {start + steps:<7} {1e6 * batch_elapsed / steps:9.1f} us/step  "
                      f"{metrics['total_spikes']:9d} batch spikes  {rate:6.2f}%")
        avg_total = elapsed * 1e6 / num_timesteps
        avg_spikes = total_spikes / num_timesteps
        speedup = num_timesteps * 0.001 / elapsed
        print("\n" + "=" * 60)
        print(f"  BENCHMARK RESULTS (averaged over {num_timesteps} timesteps)")
        print("=" * 60)
        print(f"  Total per timestep: {avg_total:8.1f} us (wall-clock, batch readback included)")
        print(f"  Avg spikes/step:    {avg_spikes:.0f} ({100 * avg_spikes / n_neurons:.2f}% firing rate)")
        print(f"  Biological time per wall-second: {speedup * 1000:.0f} ms")
        print(f"  Speed vs real-time:              {speedup:.1f}x")
        if speedup >= 1.0:
            print("  >>> FASTER THAN REAL-TIME <<<")
        else:
            print(f"  {1.0 / speedup:.1f}x slower than real-time")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FlyWire Connectome SYCL GPU Simulator")
    parser.add_argument("--data", help="Binary connectome file (from download_connectome.py)")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", help="Explicit SYCL selector; default: Intel GPU (or FASTFLY_DEVICE)")
    parser.add_argument("--neurons", type=int, default=139255, help="Synthetic neuron count")
    parser.add_argument("--synapses", type=int, default=54500000, help="Synthetic synapse count")
    args = parser.parse_args()
    if args.device is not None:
        os.environ["FASTFLY_DEVICE"] = args.device
    try:
        if args.timesteps <= 0 or args.warmup < 0:
            raise ValueError("timesteps must be positive and warmup must be nonnegative")
        print_gpu_info()
        connectome = (load_connectome_binary(args.data) if args.data else
                      generate_synthetic(args.neurons, args.synapses, args.seed))
        run_simulation(*connectome, num_timesteps=args.timesteps, warmup_steps=args.warmup,
                       seed=args.seed, verbose=args.verbose)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
