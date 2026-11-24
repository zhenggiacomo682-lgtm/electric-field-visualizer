import numpy as np
import matplotlib.pyplot as plt

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

resolutions = [50, 100, 200, 500, 1000]

grid_spacings = []
mean_errors = []
max_errors = []

for resolution in resolutions:

    X, Y = create_grid(-5, 5, -5, 5, resolution)

    h = X[0, 1] - X[0, 0]

    Ex, Ey = calculate_field_grid_vectorized(
        charges, charge_positions, X, Y
    )

    V = calculate_potential_grid(
        charges, charge_positions, X, Y
    )

    Ex_num, Ey_num = calculate_field_from_potential(
        V, X, Y
    )

    error = np.sqrt(
        (Ex - Ex_num)**2 +
        (Ey - Ey_num)**2
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

    grid_spacings.append(h)
    mean_errors.append(mean_error)
    max_errors.append(max_error)

    print(f"Resolution: {resolution} x {resolution}")
    print(f"Grid spacing: {h:.6f} m")
    print(f"Mean error: {mean_error:.8f} N/C")
    print(f"Maximum error: {max_error:.8f} N/C")
    print()

grid_spacings = np.array(grid_spacings)
mean_errors = np.array(mean_errors)
max_errors = np.array(max_errors)

order = np.polyfit(
    np.log(grid_spacings),
    np.log(mean_errors),
    1
)[0]

print(f"Estimated convergence order: {order:.4f}")

reference = mean_errors[0] * (
    grid_spacings / grid_spacings[0]
)**2

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(
    grid_spacings, mean_errors,
    "o-", label="Mean error"
)

ax.loglog(
    grid_spacings, max_errors,
    "s-", label="Maximum error"
)

ax.loglog(
    grid_spacings, reference,
    "--", label="Second-order reference"
)

ax.set_xlabel("Grid spacing h (m)")
ax.set_ylabel("Absolute error (N/C)")
ax.set_title("Numerical Convergence Analysis")

ax.grid(True, which="both", alpha=0.2)
ax.legend()

fig.tight_layout()

plt.show()
