# FastFly

GPU-accelerated simulator for the complete *Drosophila melanogaster* (fruit fly) brain connectome — 139,255 neurons and 54.5 million synapses — targeting real-time or faster performance on Intel Arc GPUs using Intel oneAPI SYCL. Performance depends on the graph and hardware; NVIDIA benchmark results from the previous implementation do not apply to Arc.


## How it works

- **Neuron model:** Leaky Integrate-and-Fire (LIF), validated by [arXiv:2404.17128](https://arxiv.org/abs/2404.17128)
- **Connectivity:** CSR sparse format; FP16 weights in the native CLI, INT8 weights with per-neuron FP32 scales in the Python/web engine
- **Spike propagation:** Push model — only processes synapses of neurons that actually fired
- **Spike detection:** Packed 32-bit spike flags, independent of GPU subgroup width
- **Load balancing:** Subgroup-per-spike push propagation with atomic FP32 accumulation
- **Runtime:** Shared SYCL backend for C++ and Python, fused noise/LIF update, batched device-side spike counting

## Data sources

This simulator uses real connectome data from the [FlyWire](https://flywire.ai/) project — a collaborative effort to map every neuron and synapse in an adult *Drosophila melanogaster* brain from electron microscopy imagery.

### Synaptic connectivity

- **Data:** `Connectivity_783.parquet` and `Completeness_783.csv` (FlyWire materialization v783)
- **Repository:** [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Paper:** Shiu PK, Sterne GR, Spiller N, et al. "A Drosophila computational brain model reveals sensorimotor processing." *Nature* 634, 210–219 (2024). [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)
- **Contents:** 139,255 neurons, 54.5M synaptic connections with signed excitatory/inhibitory weights in CSR sparse format

### Neuron annotations (cell types, positions, neurotransmitters)

- **Data:** `Supplemental_file1_neuron_annotations.tsv`
- **Repository:** [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Paper:** Schlegel P, Yin Y, Bates AS, et al. "Whole-brain annotation and multi-connectome cell typing of Drosophila." *Nature* 634, 139–152 (2024). [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)
- **Contents:** Cell type classifications (super_class, cell_class, cell_type), neurotransmitter identity, laterality, nerve assignments, and 3D soma positions in FAFB voxel coordinates (4×4×40 nm resolution)

### Underlying electron microscopy volume

- **Dataset:** FAFB (Full Adult Fly Brain)
- **Paper:** Zheng Z, Lauritzen JS, Perlman E, et al. "A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster." *Cell* 174(3), 730–743 (2018). [doi:10.1016/j.cell.2018.06.019](https://doi.org/10.1016/j.cell.2018.06.019)

### FlyWire connectome

- **Platform:** [flywire.ai](https://flywire.ai/) · [Codex browser](https://codex.flywire.ai/)
- **Paper:** Dorkenwald S, Matsliah A, Sterling AR, et al. "Neuronal wiring diagram of an adult brain." *Nature* 634, 124–138 (2024). [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)

### Neuron model validation

- Zhang X, Yang P, Feng J, et al. "Network Structure Governs Drosophila Brain Functionality." [arXiv:2404.17128](https://arxiv.org/abs/2404.17128) (2024). Demonstrates that network structure dominates over neuron model choice, validating the use of LIF for whole-brain simulation.

## Requirements

- Intel Arc GPU and a current Intel graphics/compute driver exposing a SYCL Level Zero GPU device.
- [Intel oneAPI DPC++/C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html), CMake 3.20+, and Ninja. A runtime-only oneAPI installation is not enough to build.
- Windows: Visual Studio C++ build tools and an initialized oneAPI command prompt. Linux: a supported Intel GPU compute runtime and access to the render device. WSL additionally requires working Intel GPU passthrough; installing the compiler alone does not enable it.
- Python 3.10+ and NumPy (`python -m pip install numpy`). Python uses the compiled shared library via `ctypes`; CuPy and CUDA are no longer required.
- Web visualizer: FastAPI and uvicorn (`python -m pip install fastapi "uvicorn[standard]"`).

## Quick start

### 1. Download the connectome

```bash
pip install pandas pyarrow numpy requests
python download_connectome.py
```

This downloads the FlyWire v783 data and produces `flywire_v783.bin`.

### 2. Build the shared backend and native simulator

Initialize the oneAPI environment before building **and running**. Check `sycl-ls`:
an Intel GPU should be listed under the Level Zero backend.

Windows (oneAPI command prompt):

```bat
build.bat
build\flywire_sim.exe --data flywire_v783.bin
```

Linux:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/flywire_sim --data flywire_v783.bin
```

Omit `--data` to generate a full-size synthetic graph. `build.bat debug` selects
a debug build; `build.bat clean` cleans generated build products.

### 3. Run the Python simulator

```bash
python -m pip install numpy
python flywire_sim.py --data flywire_v783.bin
python flywire_sim.py --neurons 1024 --synapses 8192 --warmup 5 --timesteps 20
```

The shared library (`fastfly_sycl.dll` or `libfastfly_sycl.so`) is discovered in
the project root, `build`, or `build/Release`. Set `FASTFLY_SYCL_LIBRARY` to its
absolute path for a custom build directory. Keep the compiler runtime libraries
on the platform library search path (the initialized oneAPI environment does this).

By default the backend requires an Intel GPU; it does **not** silently use a CPU.
To select a particular device, pass `--device level_zero:gpu:0` to either CLI,
or set `FASTFLY_DEVICE=level_zero:gpu:0` for Python/the web server. Device indices
are backend-relative; inspect `sycl-ls` on your machine.
An explicit `--device opencl:cpu` can exercise the same SYCL kernels on an
installed CPU runtime for correctness checks, but is not GPU acceleration.
Missing devices or unsupported device capabilities produce an error.

The LIF equation, noise hash, next-step synaptic propagation, binary data format,
and web metrics/stimulus interface are retained. Python initialization now uses
NumPy's seeded generator rather than CuPy's; trajectories are not bit-identical
to the previous backend. Parallel floating-point accumulation can also change
rounding and subsequent spikes.

### 4. Web visualizer

```bash
pip install fastapi uvicorn[standard]
python app_server.py --data flywire_v783.bin
# Open http://127.0.0.1:8000
```

## Correctness checks

After building and initializing the runtime, run the regression suite on the
default Intel GPU:

```bash
python -m unittest discover -s tests -v
```

For an explicitly installed SYCL OpenCL CPU runtime, the same tests can run with
`FASTFLY_DEVICE=opencl:cpu` (on Windows: `set FASTFLY_DEVICE=opencl:cpu`).
They compare actual SYCL execution against a NumPy reference for both weight
formats, partial spike words, empty connectivity, inhibitory/colliding synapses,
batch continuity, stimulus clearing, and group/motor counts.
CPU correctness checks are not evidence of Arc performance; benchmark the full
connectome on your target Arc device separately.

## Project structure

| File | Description |
|---|---|
| `fastfly_sycl.cpp`, `fastfly_sycl.h` | Shared SYCL kernels and C API |
| `flywire_sim.cpp` | Standalone oneAPI C++ simulator (FP16 weights) |
| `sycl_backend.py` | NumPy/ctypes interface to the shared SYCL backend |
| `flywire_sim.py` | Python benchmark and connectome utilities (INT8 weights) |
| `sim_engine.py` | SYCL-backed simulation engine used by the web server |
| `app_server.py` | FastAPI web server with WebSocket-based live visualizer |
| `download_connectome.py` | Downloads FlyWire v783 data and converts to binary format |
| `download_metadata.py` | Downloads neuron annotation metadata |
| `CMakeLists.txt`, `build.bat` | Cross-platform oneAPI build and Windows entry point |

## License

MIT
