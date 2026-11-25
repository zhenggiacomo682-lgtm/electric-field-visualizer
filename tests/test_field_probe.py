import numpy as np

from src.field_probe import calculate_probe_values
from src.electrostatics import COULOMB_CONSTANT


charges = np.array([5e-9, -5e-9])

charge_positions = np.array([
    [-2.0, 0.0],
    [2.0, 0.0]
])

point = np.array([0.0, 0.0])

field, magnitude, potential = calculate_probe_values(
    charges,
    charge_positions,
    point
)

expected_field = np.array([
    COULOMB_CONSTANT * 10e-9 / 4,
    0.0
])

expected_magnitude = np.linalg.norm(expected_field)

np.testing.assert_allclose(field, expected_field)
np.testing.assert_allclose(magnitude, expected_magnitude)
np.testing.assert_allclose(potential, 0.0, atol=1e-12)

print(f"Electric field: {field} N/C")
print(f"Field magnitude: {magnitude:.6f} N/C")
print(f"Electric potential: {potential:.6f} V")

print("Field probe test passed successfully.")