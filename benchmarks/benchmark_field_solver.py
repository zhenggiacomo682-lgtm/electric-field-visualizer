import time
import numpy as np
import matplotlib.pyplot as plt

from src.grid import create_grid

from src.field_solver import (
    calculate_field_grid,
    calculate_field_grid_vectorized
)
charges = np.array([2e-9, -2e-9])

charge_positions = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])

resolutions = [50, 100, 200, 500, 1000]

original_times = []
vectorized_times = []
for resolution in resolutions:

    X, Y = create_grid(-5, 5, -5, 5, resolution)

    start = time.perf_counter()

    calculate_field_grid(charges, charge_positions, X, Y)

    original_time = time.perf_counter() - start

    original_times.append(original_time)

    start = time.perf_counter()

    calculate_field_grid_vectorized(charges, charge_positions, X, Y)

    vectorized_time = time.perf_counter() - start

    vectorized_times.append(vectorized_time)

    speedup = original_time / vectorized_time

    print(f"Resolution: {resolution} x {resolution}")
    print(f"Original solver: {original_time:.6f} s")
    print(f"Vectorized solver: {vectorized_time:.6f} s")
    print(f"Speedup: {speedup:.2f}x")
    print()