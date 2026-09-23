"""
SimEngine — importable wrapper around the FlyWire SYCL simulator.

Loads neuron_annotations.npz (from download_metadata.py) for biologically
meaningful stimuli, heatmap groups, 3D positions, and motor neuron detail.

Usage (standalone test):
    python sim_engine.py                        # synthetic data
    python sim_engine.py --data flywire_v783.bin
"""

import base64
import os
import time
import sys
import threading
import numpy as np

from flywire_sim import load_connectome_binary, generate_synthetic
from sycl_backend import NativeSimulation

ANNOTATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "neuron_annotations.npz")


class SimEngine:
    """GPU-accelerated LIF simulator using the native SYCL backend."""

    def __init__(self, data_file=None, seed=42, *, n_neurons=139255,
                 n_synapses=54500000, device_selector=None):
        self._lock = threading.RLock()
        if data_file:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                load_connectome_binary(data_file)
        else:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                generate_synthetic(n_neurons, n_synapses, seed=seed)
        self.seed = seed
        self.current_step = 0
        voltage = np.random.default_rng(seed).uniform(0.0, 0.9, self.n_neurons).astype(np.float32)
        self._backend = NativeSimulation(offsets, targets, weights, voltage, seed=seed,
                                         int8_weights=True, device_selector=device_selector)
        self.tau_decay = np.float32(0.9)
        self.v_threshold = np.float32(1.0)
        self.v_reset = np.float32(0.0)
        self.noise_amp = np.float32(0.4)
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0
        self._last_spike_indices = np.array([], dtype=np.uint32)
        self.send_active_indices = True
        self.send_group_rates = True
        self.send_motor_rates = True
        self.active_indices_interval = 3
        self._batch_counter = 0
        try:
            self._load_annotations()
            self._backend.set_groups(self._neuron_to_group, self.num_groups,
                                      self._neuron_to_motor, self._num_motor_groups)
        except Exception:
            self._backend.close()
            raise
        self.group_rates_history = []

    @property
    def device_name(self):
        return self._backend.device_name

    def close(self):
        with self._lock:
            self._backend.close()

    def __enter__(self):
        self._backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _load_annotations(self):
        """Load neuron_annotations.npz for biological groups, stimuli, positions."""
        data = None
        if os.path.exists(ANNOTATIONS_FILE):
            with np.load(ANNOTATIONS_FILE, allow_pickle=True) as archive:
                data = {name: archive[name] for name in archive.files}
            for name in ("root_ids", "pos_x", "pos_y", "pos_z", "super_class"):
                if name in data and len(data[name]) != self.n_neurons:
                    print("Annotation neuron count does not match this connectome; using index groups.")
                    data = None
                    break
        if data is not None:
            print(f"Loading neuron annotations from {ANNOTATIONS_FILE}...")

            # Root IDs
            self._root_ids = data['root_ids'].astype(np.int64) if 'root_ids' in data else None

            # 3D positions (normalized to [-1,1])
            if 'pos_x' in data:
                self._positions = np.stack([
                    data['pos_x'], data['pos_y'], data['pos_z']
                ], axis=1).astype(np.float32)  # [N, 3]
                print(f"  3D positions loaded: {self._positions.shape}")
            else:
                self._positions = None

            # Super class per neuron (for coloring in 3D)
            self._super_class = data.get('super_class', None)

            # Stimuli
            self._stimuli = {}
            stim_names = list(data['stim_names'])
            for name in stim_names:
                safe = 'stim_' + name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                if safe in data:
                    self._stimuli[name] = data[safe].astype(np.int32)

            # Heatmap groups
            self.group_labels = list(data['group_names'])
            self.num_groups = len(self.group_labels)
            self._group_indices = []
            for name in self.group_labels:
                self._group_indices.append(data['group_' + name].astype(np.int32))

            self._neuron_to_group = np.full(self.n_neurons, -1, dtype=np.int32)
            for g, indices in enumerate(self._group_indices):
                self._neuron_to_group[indices] = g

            # Body sensory groups
            self._body_sensory = {}
            if 'body_sensory_names' in data:
                for name in data['body_sensory_names']:
                    key = 'bsens_' + name
                    if key in data:
                        self._body_sensory[str(name)] = data[key].astype(np.int32)

            # Body motor groups
            self._body_motor = {}
            if 'body_motor_names' in data:
                motor_names = [str(n) for n in data['body_motor_names']]
                for name in motor_names:
                    key = 'bmotor_' + name
                    if key in data:
                        self._body_motor[name] = data[key].astype(np.int32)

            self._motor_group_names = list(self._body_motor.keys())
            self._num_motor_groups = len(self._motor_group_names)
            self._neuron_to_motor = np.full(self.n_neurons, -1, dtype=np.int32)
            for g, name in enumerate(self._motor_group_names):
                indices = self._body_motor[name]
                self._neuron_to_motor[indices] = g

            self._use_annotations = True
            print(f"  {len(self._stimuli)} stimuli, {self.num_groups} heatmap groups")
            print(f"  {len(self._body_sensory)} body sensory, {len(self._body_motor)} body motor")

        else:
            print(f"No matching annotation file available ({ANNOTATIONS_FILE})")
            print("  Run download_metadata.py for biological annotations.")
            self._root_ids = None
            self._positions = None
            self._super_class = None
            self._setup_fallback_groups()
            self._use_annotations = False

    def _setup_fallback_groups(self):
        """Fallback: equal-size index-range groups."""
        self.num_groups = 20
        group_size = self.n_neurons // self.num_groups
        self.group_labels = [f"Group {i}" for i in range(self.num_groups)]
        self._group_indices = []
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else self.n_neurons
            self._group_indices.append(np.arange(start, end, dtype=np.int32))

        self._neuron_to_group = np.full(self.n_neurons, -1, dtype=np.int32)
        for g, indices in enumerate(self._group_indices):
            self._neuron_to_group[indices] = g

        self._stimuli = {
            "Neurons 0-1000": np.arange(0, min(1000, self.n_neurons), dtype=np.int32),
        }
        self._body_sensory = {}
        self._body_motor = {}
        self._motor_group_names = []
        self._num_motor_groups = 0
        self._neuron_to_motor = np.full(self.n_neurons, -1, dtype=np.int32)

    def inject_stimulus(self, neuron_indices, amplitude=0.5):
        with self._lock:
            self._backend.set_stimulus(neuron_indices, amplitude)
            self._stimulus_indices = np.unique(np.asarray(neuron_indices, dtype=np.uint32))
            self._stimulus_amplitude = float(amplitude)

    def clear_stimulus(self):
        with self._lock:
            self._backend.clear_stimulus()
            self._stimulus_indices = None
            self._stimulus_amplitude = 0.0

    def set_noise_amp(self, value):
        value = np.float32(value)
        if not np.isfinite(value):
            raise ValueError("Noise amplitude must be finite")
        with self._lock:
            self.noise_amp = value

    def step(self, n=50):
        """Run a native batch with GPU-side counts and no per-substep readback."""
        with self._lock:
            return self._step(n)

    def _step(self, n):
        interval = self.active_indices_interval
        if not isinstance(interval, (int, np.integer)) or interval < 1:
            raise ValueError("active_indices_interval must be a positive integer")
        t_start = time.perf_counter()
        self._backend.step(n, decay=self.tau_decay, threshold=self.v_threshold,
                           reset=self.v_reset, noise=self.noise_amp)
        native = self._backend.read_metrics()
        self.current_step += n
        total_spikes = native["total_spikes"]
        t_elapsed = time.perf_counter() - t_start
        result = {
            "step": self.current_step,
            "spike_count": total_spikes,
            "firing_rate": round(total_spikes / (n * self.n_neurons), 6),
            "mean_voltage": round(native["mean_voltage"], 4),
            "steps_per_sec": round(n / t_elapsed, 1) if t_elapsed > 0 else 0,
        }
        if self.send_group_rates:
            group_rates = []
            for g, indices in enumerate(self._group_indices):
                group_n = len(indices)
                rate = float(native["group_counts"][g]) / (n * group_n) if group_n else 0
                group_rates.append(round(rate, 6))
            self.group_rates_history.append(group_rates)
            if len(self.group_rates_history) > 200:
                del self.group_rates_history[:-200]
            result["group_rates"] = group_rates
        if self.send_motor_rates:
            motor_rates = {}
            for g, name in enumerate(self._motor_group_names):
                group_n = len(self._body_motor[name])
                rate = float(native["motor_counts"][g]) / (n * group_n) if group_n else 0
                motor_rates[name] = round(rate, 6)
            result["motor_rates"] = motor_rates
        self._batch_counter += 1
        if self.send_active_indices and self._batch_counter % interval == 0:
            self._last_spike_indices = self._backend.read_spikes()
            result["active_indices"] = self._last_spike_indices.tolist()
        return result

    # --- Data accessors ---

    def get_predefined_stimuli(self):
        return list(self._stimuli.keys())

    def get_body_info(self):
        sensory = {k: len(v) for k, v in self._body_sensory.items()}
        motor = {k: len(v) for k, v in self._body_motor.items()}
        return {"sensory": sensory, "motor": motor}

    def get_positions_b64(self):
        """Return neuron positions as base64-encoded float32 array [N*3]."""
        if self._positions is not None:
            return base64.b64encode(self._positions.tobytes()).decode('ascii')
        return None

    def get_neuron_classes(self):
        """Return super_class per neuron for 3D coloring."""
        if self._super_class is not None:
            # Encode as int: unique classes -> color indices
            unique = sorted(set(self._super_class))
            class_to_id = {c: i for i, c in enumerate(unique)}
            ids = np.array([class_to_id.get(c, 0) for c in self._super_class],
                           dtype=np.uint8)
            return {
                "labels": unique,
                "ids_b64": base64.b64encode(ids.tobytes()).decode('ascii')
            }
        return None

    def get_motor_detail(self, group_name):
        """Return detail for a motor group: neuron indices, root_ids, active status."""
        if group_name not in self._body_motor:
            return None
        indices = self._body_motor[group_name]
        active_set = set(self._last_spike_indices.tolist())
        neurons = []
        for idx in indices:
            idx = int(idx)
            rid = int(self._root_ids[idx]) if self._root_ids is not None else idx
            neurons.append({
                "index": idx,
                "root_id": rid,
                "active": idx in active_set,
            })
        return {"group": group_name, "neurons": neurons}

    def apply_predefined_stimulus(self, name, amplitude=None):
        if name not in self._stimuli:
            return False
        indices = self._stimuli[name]
        amp = amplitude if amplitude is not None else 0.5
        self.inject_stimulus(indices, amp)
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FlyWire SYCL simulation engine")
    parser.add_argument("--data", help="Binary connectome file")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--device", help="Explicit SYCL selector (or FASTFLY_DEVICE)")
    parser.add_argument("--neurons", type=int, default=139255)
    parser.add_argument("--synapses", type=int, default=54500000)
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        parser.error("--steps and --batch must be positive")
    try:
        with SimEngine(data_file=args.data, n_neurons=args.neurons, n_synapses=args.synapses,
                       device_selector=args.device) as engine:
            print(f"\nSimEngine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses")
            print(f"SYCL device: {engine.device_name}")
            print(f"Stimuli: {engine.get_predefined_stimuli()}")
            print(f"Groups:  {engine.group_labels}")
            print(f"Positions: {'yes' if engine._positions is not None else 'no'}")
            print(f"Running {args.steps} steps in batches of {args.batch}...\n")
            for i in range(0, args.steps, args.batch):
                metrics = engine.step(n=min(args.batch, args.steps - i))
                print(f"  Step {metrics['step']:>6d}  "
                      f"spikes={metrics['spike_count']:>6d}  "
                      f"rate={metrics['firing_rate']*100:>5.2f}%  "
                      f"V_mean={metrics['mean_voltage']:.3f}  "
                      f"active_3d={len(metrics.get('active_indices', []))}  "
                      f"steps/s={metrics['steps_per_sec']:.0f}")
        print("\nDone.")
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
