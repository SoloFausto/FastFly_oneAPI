#ifndef FASTFLY_SYCL_H
#define FASTFLY_SYCL_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(FASTFLY_SYCL_BUILD)
#    define FF_API __declspec(dllexport)
#  else
#    define FF_API __declspec(dllimport)
#  endif
#else
#  define FF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Errors are thread-local, valid until the next API call on that thread.
 * Handles must not be accessed concurrently. Host buffers need only survive
 * their call. All status functions return 0 on success and -1 on failure. */
FF_API const char* ff_last_error(void);
FF_API void* ff_create(uint32_t n, uint32_t s, const uint32_t* offsets,
                       const uint32_t* targets, const float* weights,
                       const float* voltage, uint32_t seed, int int8_weights,
                       const char* device_selector);
FF_API void ff_destroy(void* handle);
FF_API const char* ff_device_name(void* handle);
/* Maps have n entries, with -1 excluding a neuron from a category. */
FF_API int ff_set_groups(void* handle, const int32_t* group_map, uint32_t groups,
                         const int32_t* motor_map, uint32_t motors);
FF_API int ff_set_stimulus(void* handle, const uint32_t* unique_indices,
                           uint32_t count, float amplitude);
/* Counters reset per batch; step/noise state and synaptic input persist.
 * Synchronizes once at the end, never between substeps. */
FF_API int ff_step(void* handle, uint32_t steps, float decay, float threshold,
                   float reset, float noise);
FF_API int ff_read_metrics(void* handle, uint64_t* total, uint64_t* groups,
                           uint64_t* motors, float* mean_voltage);
/* indices must reserve n entries; returns spikes from the last substep. */
FF_API int ff_read_spikes(void* handle, uint32_t* indices, uint32_t* count);
FF_API int ff_read_state(void* handle, float* voltage, float* current);

#ifdef __cplusplus
}
#endif
#endif
