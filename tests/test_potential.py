import numpy as np

from src.field_solver import calculate_potential_grid
from src.electrostatics import COULOMB_CONSTANT

charges = np.array([2e-9, -2e-9])

charge_positions = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])

X = np.array([[0.0, 2.0]])
Y = np.array([[0.0, 0.0]])

V = calculate_potential_grid(
    charges,
    charge_positions,
    X,
    Y
)

expected = np.array([[
    0.0,
    COULOMB_CONSTANT * (2e-9 / 3 - 2e-9)
]])

np.testing.assert_allclose(V, expected, atol=1e-10)

print("Potential solver test passed successfully.")
