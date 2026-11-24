import numpy as np

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

X, Y = create_grid(-5, 5, -5, 5, 101)

Ex_original, Ey_original = calculate_field_grid(
    charges, charge_positions, X, Y
)

Ex_vectorized, Ey_vectorized = calculate_field_grid_vectorized(
    charges, charge_positions, X, Y
)

np.testing.assert_allclose(
    Ex_original, Ex_vectorized,
    rtol=1e-7,
    atol=1e-10,
    equal_nan=True
)

np.testing.assert_allclose(
    Ey_original, Ey_vectorized,
    rtol=1e-7,
    atol=1e-10,
    equal_nan=True
)

print("Vectorized field solver test passed successfully.")