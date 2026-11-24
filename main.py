import numpy as np

from src.grid import create_grid
from src.field_solver import calculate_field_grid
from src.visualization import plot_electric_field

print("=== ELECTRIC FIELD SIMULATOR ===")
charges = np.array([2e-9, -2e-9])
charge_positions = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])

X, Y = create_grid(-5, 5, -5, 5, 200)

Ex, Ey = calculate_field_grid(
    charges,
    charge_positions,
    X,
    Y
)
plot_electric_field(
    X, Y, Ex, Ey,
    charges,
    charge_positions
)