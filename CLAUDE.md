# FlyWire Connectome GPU Simulator

Intel oneAPI SYCL simulator for Intel Arc GPUs targeting real-time (or faster) simulation of the complete
Drosophila melanogaster brain connectome (139,255 neurons, 54.5M synapses).

## Architecture

- **Neuron model**: Leaky Integrate-and-Fire (LIF), validated as sufficient by arXiv:2404.17128
- **Spike detection**: Packed 32-bit spike flags independent of native subgroup width
- **Spike propagation**: Push model - only process synapses of neurons that actually fired
- **Connectivity**: CSR sparse format; FP16 native CLI weights and per-neuron scaled INT8 Python/web weights
- **Load balancing**: Subgroup-per-spike with grid-stride loop
- **Integration**: Shared SYCL C API used by the native CLI and Python ctypes bridge

## Build

Requires Intel oneAPI DPC++/C++, an Intel GPU compute driver, CMake, and Ninja.
Initialize the oneAPI environment, then run `build.bat` on Windows. On Linux:
`cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release`
and `cmake --build build --parallel`. See README.md for device selection.

## Key optimization targets

1. Profile spike propagation on the target Arc GPU (memory traffic and atomic contention)
2. At 1% firing rate: ~549K active synapses/step; ~3.3 MB FP16 or ~2.7 MB INT8 target/weight traffic, excluding scales and accumulation
3. Keep spike counts device-side during batches; measure actual wall time before claiming real-time performance

## Next optimizations to try

- Shared-memory accumulation buffers to reduce atomicAdd contention
- Connectivity pruning (remove weakest synapses)
- Subgroup-cooperative target sorting to batch atomic additions
- Multi-stream pipelining
- Pull model comparison at higher firing rates
