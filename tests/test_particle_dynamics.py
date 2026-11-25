import numpy as np

from src.particle_dynamics import (
    particle_derivative,
    simulate_euler
)

charges = np.array([5e-9, -5e-9])

charge_positions = np.array([
    [-2.0, 0.0],
    [2.0, 0.0]
])

initial_state = np.array([
    0.0, 0.0, 0.0, 0.0
])

particle_charge = 1e-9
particle_mass = 1e-9

derivative = particle_derivative(
    initial_state,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
)

expected_derivative = np.array([
    0.0, 0.0, 22.46887948075, 0.0
])

np.testing.assert_allclose(
    derivative,
    expected_derivative
)

states = simulate_euler(
    initial_state,
    0.01,
    100,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
)

assert states.shape == (101, 4)

np.testing.assert_allclose(
    states[1],
    initial_state + 0.01 * expected_derivative
)

print("Initial derivative:", derivative)
print("Final state:", states[-1])
print("Particle dynamics test passed successfully.")
