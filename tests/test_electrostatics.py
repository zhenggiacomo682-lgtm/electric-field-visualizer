import numpy as np
from src.electrostatics import electric_field, total_electric_field

q = 2e-9

charge_position = np.array([0.0, 0.0])
observation_point = np.array([3.0, 4.0])

result = electric_field(q, charge_position, observation_point)

expected = np.array([0.43140249, 0.57520331])

np.testing.assert_allclose(result, expected, rtol=1e-7)

print("Test passed successfully.")

charges = np.array([2e-9, -2e-9])
charge_positions = np.array([
    [-1.0, 0.0],
    [1.0, 0.0]
])
observation_point = np.array([0.0, 0.0])
result = total_electric_field(charges, charge_positions, observation_point)

expected = np.array([35.9502071692, 0.0])

np.testing.assert_allclose(result, expected, rtol=1e-7)

print("Multiple charges test passed successfully.")