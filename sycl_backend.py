"""ctypes interface to the shared FastFly oneAPI SYCL backend.

Importing this module does not load the library or select a device. By default
creation requires an Intel GPU; FASTFLY_DEVICE explicitly overrides selection.
"""

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import threading
import weakref

import numpy as np


_U32 = ct.POINTER(ct.c_uint32)
_I32 = ct.POINTER(ct.c_int32)
_U64 = ct.POINTER(ct.c_uint64)
_F32 = ct.POINTER(ct.c_float)


def _pointer(array, pointer_type):
    return array.ctypes.data_as(pointer_type)


@lru_cache(maxsize=None)
def _load_library(override):
    root = Path(__file__).resolve().parent
    names = ("fastfly_sycl.dll",) if os.name == "nt" else ("libfastfly_sycl.so",)
    candidates = ([Path(override).expanduser()] if override else
                  [directory / name for directory in (root, root / "build", root / "build" / "Release")
                   for name in names])
    errors = []
    dll_directories = []
    if os.name == "nt":
        # Python 3.8+ does not search PATH for extension DLL dependencies.
        # Honor the initialized oneAPI environment and keep these handles alive
        # for backend/plugin DLLs loaded after the shared library itself.
        for directory in dict.fromkeys(os.environ.get("PATH", "").split(os.pathsep)):
            directory = directory.strip('"')
            if directory and Path(directory).is_dir():
                try:
                    dll_directories.append(os.add_dll_directory(str(Path(directory).resolve())))
                except OSError as error:
                    errors.append(f"DLL directory {directory}: {error}")
    for path in candidates:
        try:
            lib = ct.CDLL(str(path.resolve()))
            break
        except OSError as error:
            errors.append(f"{path}: {error}")
    else:
        for directory in dll_directories:
            directory.close()
        raise RuntimeError(
            "Cannot load the FastFly SYCL library. Build the fastfly_sycl shared target "
            "with Intel oneAPI and initialize the oneAPI runtime environment. Set "
            "FASTFLY_SYCL_LIBRARY to its full path if installed elsewhere.\n" + "\n".join(errors)
        )
    signatures = {
        "ff_last_error": ([], ct.c_char_p),
        "ff_create": ([ct.c_uint32, ct.c_uint32, _U32, _U32, _F32, _F32,
                       ct.c_uint32, ct.c_int, ct.c_char_p], ct.c_void_p),
        "ff_destroy": ([ct.c_void_p], None),
        "ff_device_name": ([ct.c_void_p], ct.c_char_p),
        "ff_set_groups": ([ct.c_void_p, _I32, ct.c_uint32, _I32, ct.c_uint32], ct.c_int),
        "ff_set_stimulus": ([ct.c_void_p, _U32, ct.c_uint32, ct.c_float], ct.c_int),
        "ff_step": ([ct.c_void_p, ct.c_uint32, ct.c_float, ct.c_float, ct.c_float, ct.c_float], ct.c_int),
        "ff_read_metrics": ([ct.c_void_p, _U64, _U64, _U64, _F32], ct.c_int),
        "ff_read_spikes": ([ct.c_void_p, _U32, _U32], ct.c_int),
        "ff_read_state": ([ct.c_void_p, _F32, _F32], ct.c_int),
    }
    try:
        for name, (argtypes, restype) in signatures.items():
            function = getattr(lib, name)
            function.argtypes = argtypes
            function.restype = restype
    except AttributeError as error:
        for directory in dll_directories:
            directory.close()
        raise RuntimeError(f"Incompatible FastFly SYCL library at {path}; rebuild it: {error}") from error
    lib._dll_directories = dll_directories
    return lib


def _uint32(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be between {minimum} and 4294967295")
    return value


def _indices(values, name, upper):
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.size and (np.any(array < 0) or np.any(array >= upper)):
        raise ValueError(f"{name} contains an out-of-range index")
    return np.ascontiguousarray(array, dtype=np.uint32)


class NativeSimulation:
    """Owned native simulation; all operations are serialized across threads.

    ``offsets, targets, weights`` are CSR arrays. ``voltage`` is initial FP32
    neuron voltage. Inputs are uploaded during construction and not retained.
    ``read_metrics()`` returns counts for the most recent ``step()`` batch;
    ``read_spikes()`` returns indices from its final timestep. ``read_state()``
    returns independent NumPy ``(voltage, current)`` arrays.
    """

    def __init__(self, offsets, targets, weights, voltage, seed=42,
                 int8_weights=True, device_selector=None):
        self._lock = threading.RLock()
        self._handle = None
        voltage = np.ascontiguousarray(voltage, dtype=np.float32)
        if voltage.ndim != 1:
            raise ValueError("voltage must be one-dimensional")
        self.n_neurons = _uint32(voltage.size, "neuron count", minimum=1)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        self.n_synapses = _uint32(weights.size, "synapse count")
        offsets = _indices(offsets, "offsets", self.n_synapses + 1)
        targets = _indices(targets, "targets", self.n_neurons)
        if (offsets.size != self.n_neurons + 1 or offsets[0] != 0 or
                offsets[-1] != self.n_synapses or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("offsets must describe monotonic CSR rows spanning all weights")
        if targets.size != self.n_synapses:
            raise ValueError("targets and weights must have equal lengths")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(voltage)):
            raise ValueError("weights and voltage must be finite")
        seed = _uint32(seed, "seed")
        selector = os.environ.get("FASTFLY_DEVICE", "") if device_selector is None else device_selector
        if not isinstance(selector, str) or "\0" in selector:
            raise ValueError("device_selector must be a string without NUL characters")
        self._lib = _load_library(os.environ.get("FASTFLY_SYCL_LIBRARY", ""))
        self._handle = self._lib.ff_create(
            self.n_neurons, self.n_synapses, _pointer(offsets, _U32), _pointer(targets, _U32),
            _pointer(weights, _F32), _pointer(voltage, _F32), seed, int(bool(int8_weights)),
            selector.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(
                f"SYCL device initialization failed: {self._error()}. "
                "The default requires an Intel GPU. Install its driver and oneAPI runtime; "
                "use FASTFLY_DEVICE or --device for an explicit SYCL selector "
                "(for example opencl:cpu for verification).")
        self._finalizer = weakref.finalize(self, self._lib.ff_destroy, self._handle)
        self._group_counts = np.empty(0, dtype=np.uint64)
        self._motor_counts = np.empty(0, dtype=np.uint64)
        self._spikes = np.empty(self.n_neurons, dtype=np.uint32)
        self._total = ct.c_uint64()
        self._mean = ct.c_float()

    def _error(self):
        error = self._lib.ff_last_error()
        return error.decode("utf-8", errors="replace") if error else "unknown native error"

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Simulation is closed")

    def _check(self, status):
        if status != 0:
            raise RuntimeError(self._error())

    @property
    def device_name(self):
        with self._lock:
            self._require_open()
            name = self._lib.ff_device_name(self._handle)
            if not name:
                raise RuntimeError(self._error())
            return name.decode("utf-8", errors="replace")

    def set_groups(self, group_map, groups, motor_map=None, motors=0):
        groups = _uint32(groups, "group count")
        motors = _uint32(motors, "motor count")
        if motor_map is None:
            motor_map = np.full(self.n_neurons, -1, dtype=np.int32)
        maps = []
        for values, count in ((group_map, groups), (motor_map, motors)):
            array = np.asarray(values)
            if (array.shape != (self.n_neurons,) or array.dtype.kind not in "iu" or
                    np.any(array < -1) or np.any(array >= count)):
                raise ValueError("Group maps must contain one integer per neuron in [-1, group count)")
            if count > 0x7FFFFFFF:
                raise ValueError("Group counts must fit signed 32-bit maps")
            maps.append(np.ascontiguousarray(array, dtype=np.int32))
        with self._lock:
            self._require_open()
            self._check(self._lib.ff_set_groups(self._handle, _pointer(maps[0], _I32), groups,
                                                _pointer(maps[1], _I32), motors))
            self._group_counts = np.empty(groups, dtype=np.uint64)
            self._motor_counts = np.empty(motors, dtype=np.uint64)

    def set_stimulus(self, indices, amplitude=0.5):
        indices = np.unique(_indices(indices, "stimulus indices", self.n_neurons))
        amplitude = float(amplitude)
        if not np.isfinite(amplitude):
            raise ValueError("Stimulus amplitude must be finite")
        with self._lock:
            self._require_open()
            self._check(self._lib.ff_set_stimulus(self._handle, _pointer(indices, _U32),
                                                 indices.size, amplitude))

    def clear_stimulus(self):
        self.set_stimulus([], 0.0)

    def step(self, steps=1, decay=0.9, threshold=1.0, reset=0.0, noise=0.4):
        steps = _uint32(steps, "steps", minimum=1)
        parameters = tuple(float(value) for value in (decay, threshold, reset, noise))
        if not all(np.isfinite(value) for value in parameters):
            raise ValueError("LIF parameters must be finite")
        with self._lock:
            self._require_open()
            self._check(self._lib.ff_step(self._handle, steps, *parameters))

    def read_metrics(self):
        with self._lock:
            self._require_open()
            self._check(self._lib.ff_read_metrics(
                self._handle, ct.byref(self._total), _pointer(self._group_counts, _U64),
                _pointer(self._motor_counts, _U64), ct.byref(self._mean)))
            return {"total_spikes": self._total.value, "group_counts": self._group_counts.copy(),
                    "motor_counts": self._motor_counts.copy(), "mean_voltage": self._mean.value}

    def read_spikes(self):
        with self._lock:
            self._require_open()
            count = ct.c_uint32()
            self._check(self._lib.ff_read_spikes(self._handle, _pointer(self._spikes, _U32), ct.byref(count)))
            return self._spikes[:count.value].copy()

    def read_state(self):
        with self._lock:
            self._require_open()
            voltage = np.empty(self.n_neurons, dtype=np.float32)
            current = np.empty(self.n_neurons, dtype=np.float32)
            self._check(self._lib.ff_read_state(self._handle, _pointer(voltage, _F32), _pointer(current, _F32)))
            return voltage, current

    def close(self):
        with self._lock:
            if self._handle:
                self._handle = None
                self._finalizer()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
