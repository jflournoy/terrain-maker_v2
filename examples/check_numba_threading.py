"""
Check numba threading configuration and test parallel execution.

This script helps diagnose whether numba parallel JIT is working correctly.
"""

import numpy as np
import time

try:
    from numba import jit, prange, get_num_threads, config
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    print("ERROR: numba not installed")
    exit(1)

print("Numba Threading Configuration")
print("=" * 60)
print(f"Numba version: {numba.__version__}")
print(f"Number of threads: {get_num_threads()}")
print(f"Threading layer: {config.THREADING_LAYER}")
print(f"Threading layer priority: {config.THREADING_LAYER_PRIORITY}")
print()

# Show environment variables that affect threading
import os
env_vars = [
    'NUMBA_NUM_THREADS',
    'NUMBA_THREADING_LAYER',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
]

print("Environment Variables:")
print("-" * 60)
for var in env_vars:
    value = os.environ.get(var, '<not set>')
    print(f"  {var}: {value}")
print()

# Test parallel execution
print("Testing Parallel Execution:")
print("-" * 60)

@jit(nopython=True, parallel=True, cache=False)
def parallel_sum(arr):
    """Sum array elements in parallel."""
    total = 0
    for i in prange(len(arr)):
        total += arr[i]
    return total

@jit(nopython=True, parallel=False, cache=False)
def serial_sum(arr):
    """Sum array elements serially."""
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Create test array
size = 100_000_000  # 100M elements
print(f"Test array size: {size:,} elements")
arr = np.ones(size, dtype=np.float64)

# Warmup and compile
print("\nWarming up (compiling JIT functions)...")
_ = parallel_sum(arr[:1000])
_ = serial_sum(arr[:1000])

# Time parallel version
print("\nTiming parallel version...")
start = time.time()
result_parallel = parallel_sum(arr)
parallel_time = time.time() - start
print(f"  Result: {result_parallel:.0f}")
print(f"  Time: {parallel_time:.3f} seconds")

# Time serial version
print("\nTiming serial version...")
start = time.time()
result_serial = serial_sum(arr)
serial_time = time.time() - start
print(f"  Result: {result_serial:.0f}")
print(f"  Time: {serial_time:.3f} seconds")

# Calculate speedup
speedup = serial_time / parallel_time
print()
print(f"Speedup: {speedup:.2f}x")
print()

# Interpret results
if speedup > 1.5:
    print("✅ Parallel execution is working! Good speedup detected.")
elif speedup > 1.0:
    print("⚠️  Parallel execution is working but speedup is minimal.")
    print("   This might be normal for small workloads or I/O bound tasks.")
else:
    print("❌ No speedup detected. Parallel execution may not be working.")
    print("   Try setting: export NUMBA_NUM_THREADS=<num_cores>")

print()
print("Recommendations:")
print("-" * 60)
if get_num_threads() == 1:
    print("❌ Only 1 thread detected!")
    print()
    print("To enable multi-threading, set environment variables:")
    print()
    print("  # For bash/zsh:")
    print(f"  export NUMBA_NUM_THREADS={os.cpu_count()}")
    print("  export NUMBA_THREADING_LAYER=tbb  # or 'omp' or 'workqueue'")
    print()
    print("  # Then run your script:")
    print("  python examples/validate_flow_complete.py ...")
    print()
    print("  # Or inline:")
    print(f"  NUMBA_NUM_THREADS={os.cpu_count()} python examples/validate_flow_complete.py ...")
else:
    print(f"✅ Using {get_num_threads()} threads")
    print("   This should provide good parallel performance.")

print()
print(f"System CPU count: {os.cpu_count()} cores")
