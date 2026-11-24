import numpy as np

from src.grid import create_grid
from src.field_solver import (
    calculate_field_grid_vectorized,
    calculate_potential_grid
)
from src.numerical_methods import calculate_field_from_potential

charges = np.array([2e-9, -2e-9])

charge_positions = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])

X, Y = create_grid(-5, 5, -5, 5, 200)

Ex, Ey = calculate_field_grid_vectorized(
    charges, charge_positions, X, Y
)

V = calculate_potential_grid(
    charges, charge_positions, X, Y
)

Ex_numerical, Ey_numerical = calculate_field_from_potential(
    V, X, Y
)

error = np.sqrt(
    (Ex - Ex_numerical)**2 +
    (Ey - Ey_numerical)**2
)

mask = np.isfinite(error)

for position in charge_positions:

    distance = np.sqrt(
        (X - position[0])**2 +
        (Y - position[1])**2
    )

    mask &= distance > 0.5

mean_error = np.mean(error[mask])
max_error = np.max(error[mask])

print(f"Mean absolute error: {mean_error:.6f} N/C")
print(f"Maximum absolute error: {max_error:.6f} N/C")
