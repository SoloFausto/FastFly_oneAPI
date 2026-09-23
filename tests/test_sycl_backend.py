"""Run against a real SYCL device: FASTFLY_DEVICE=opencl:cpu python -m unittest discover -s tests."""
import unittest

import numpy as np

from flywire_sim import quantize_weights_int8
from sycl_backend import NativeSimulation


def reference_step(voltage, current, offsets, targets, weights, seed, step, stimulus, amplitude, noise):
    current[stimulus] += np.float32(amplitude)
    for i in range(len(voltage)):
        h = (i ^ ((step * 2654435761) & 0xFFFFFFFF) ^ ((seed * 1664525) & 0xFFFFFFFF))
        h ^= h >> 16
        h = (h * 0x45D9F3B) & 0xFFFFFFFF
        h ^= h >> 16
        current[i] += np.float32(noise) * np.float32((h & 0xFFFF) / 32768.0 - 1.0)
    voltage[:] = voltage * np.float32(0.9) + current
    spikes = np.flatnonzero(voltage >= 1.0)
    voltage[spikes] = 0
    current.fill(0)
    for i in spikes:
        np.add.at(current, targets[offsets[i]:offsets[i + 1]], weights[offsets[i]:offsets[i + 1]])
    return spikes


class SyclBehaviorTests(unittest.TestCase):
    def test_propagation_counts_and_batch_continuity(self):
        # Partial spike word, multiple workgroups, trailing empty CSR rows,
        # inhibitory weights and colliding targets all exercise GPU boundaries.
        n = 257
        offsets = np.concatenate((np.arange(0, 513, 2, dtype=np.uint32), np.array([512], dtype=np.uint32)))
        targets = np.tile(np.array([0, 256], dtype=np.uint32), 256)
        weights = np.tile(np.array([0.012, -0.007], dtype=np.float32), 256)
        weights[0:2] = [0.25, 0.125]
        initial = np.linspace(0.8, 1.2, n, dtype=np.float32)
        group_map = (np.arange(n) % 3).astype(np.int32)
        group_map[::7] = -1
        motor_map = np.full(n, -1, dtype=np.int32)
        motor_map[[0, 32, 256]] = [0, 1, 0]
        stimulus = np.array([0, 32, 256], dtype=np.uint32)
        seed = 42
        for int8_weights in (False, True):
            with self.subTest(int8_weights=int8_weights):
                if int8_weights:
                    quantized, scales = quantize_weights_int8(weights, offsets, n)
                    effective = quantized.astype(np.float32) * np.repeat(scales, np.diff(offsets))
                else:
                    effective = weights.astype(np.float16).astype(np.float32)
                voltage, current = initial.copy(), np.zeros(n, dtype=np.float32)
                with NativeSimulation(offsets, targets, weights, initial, seed=seed, int8_weights=int8_weights) as sim:
                    sim.set_groups(group_map, 3, motor_map, 2)
                    sim.set_stimulus(stimulus, 0.3)
                    step = 0
                    for batch, stimulated in ((4, True), (3, False)):
                        if not stimulated:
                            sim.clear_stimulus()
                        totals = 0
                        groups = np.zeros(3, dtype=np.uint64)
                        motors = np.zeros(2, dtype=np.uint64)
                        for _ in range(batch):
                            spikes = reference_step(voltage, current, offsets, targets, effective, seed, step,
                                                    stimulus if stimulated else np.array([], dtype=np.uint32), 0.3, 0.4)
                            totals += len(spikes)
                            for neuron in spikes:
                                if group_map[neuron] >= 0:
                                    groups[group_map[neuron]] += 1
                                if motor_map[neuron] >= 0:
                                    motors[motor_map[neuron]] += 1
                            step += 1
                        sim.step(batch, noise=0.4)
                        actual_voltage, actual_current = sim.read_state()
                        np.testing.assert_allclose(actual_voltage, voltage, atol=2e-5, rtol=2e-5)
                        np.testing.assert_allclose(actual_current, current, atol=2e-5, rtol=2e-5)
                        np.testing.assert_array_equal(np.sort(sim.read_spikes()), spikes)
                        metrics = sim.read_metrics()
                        self.assertEqual(metrics['total_spikes'], totals)
                        np.testing.assert_array_equal(metrics['group_counts'], groups)
                        np.testing.assert_array_equal(metrics['motor_counts'], motors)
                        self.assertAlmostEqual(metrics['mean_voltage'], float(voltage.mean()), places=5)

    def test_empty_connectivity_and_partial_spike_word(self):
        n = 33
        for int8_weights in (False, True):
            with self.subTest(int8_weights=int8_weights), NativeSimulation(
                np.zeros(n + 1, dtype=np.uint32), np.array([], dtype=np.uint32),
                np.array([], dtype=np.float32), np.ones(n, dtype=np.float32),
                int8_weights=int8_weights,
            ) as sim:
                sim.set_stimulus([0, 32], 1.0)
                sim.step(2, noise=0.0)
                np.testing.assert_array_equal(np.sort(sim.read_spikes()), [0, 32])
                self.assertEqual(sim.read_metrics()['total_spikes'], 4)
                sim.clear_stimulus()
                sim.step(1, noise=0.0)
                self.assertEqual(sim.read_metrics()['total_spikes'], 0)
                np.testing.assert_array_equal(sim.read_state()[1], np.zeros(n))


if __name__ == '__main__':
    unittest.main()
