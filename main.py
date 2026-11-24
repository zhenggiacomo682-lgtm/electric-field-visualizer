import numpy as np
import matplotlib.pyplot as plt

from src.grid import create_grid
from src.field_solver import (
    calculate_field_grid_vectorized,
    calculate_potential_grid
)
from src.visualization import (
    plot_electric_field,
    plot_electric_potential
)
from src.input_handler import get_positive_integer, get_float

print("=== ELECTRIC FIELD SIMULATOR ===")

number_of_charges = get_positive_integer(
    "Enter the number of charges: "
)

charges = []
charge_positions = []

for i in range(number_of_charges):

    print(f"\nCharge {i + 1}")

    q = get_float("Enter the charge (nC): ")
    x = get_float("Enter the x-coordinate (m): ")
    y = get_float("Enter the y-coordinate (m): ")

    charges.append(q * 1e-9)
    charge_positions.append([x, y])

charges = np.array(charges)
charge_positions = np.array(charge_positions)

X, Y = create_grid(-5, 5, -5, 5, 200)

Ex, Ey = calculate_field_grid_vectorized(
    charges,
    charge_positions,
    X,
    Y
)

V = calculate_potential_grid(
    charges,
    charge_positions,
    X,
    Y
)

plot_electric_field(
    X, Y, Ex, Ey, V,
    charges,
    charge_positions
)

plot_electric_potential(
    X, Y, V,
    charges,
    charge_positions
)

plt.show()