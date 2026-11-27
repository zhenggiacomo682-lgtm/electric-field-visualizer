import numpy as np

from src.energy import calculate_particle_energy
from src.electrostatics import COULOMB_CONSTANT


charges = np.array([-5e-9])

charge_positions = np.array([
    [0.0, 0.0]
])

particle_charge = 1e-9
particle_mass = 1e-9

radius = 2.0

velocity = np.sqrt(
    COULOMB_CONSTANT * abs(charges[0] * particle_charge)
    / (particle_mass * radius)
)

state = np.array([
    radius,
    0.0,
    0.0,
    velocity
])

kinetic, potential, total = calculate_particle_energy(
    state,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
)

expected_potential = (
    COULOMB_CONSTANT * charges[0] * particle_charge
    / radius
)

expected_kinetic = -0.5 * expected_potential
expected_total = 0.5 * expected_potential

np.testing.assert_allclose(kinetic, expected_kinetic)
np.testing.assert_allclose(potential, expected_potential)
np.testing.assert_allclose(total, expected_total)

print(f"Kinetic energy: {kinetic:.10e} J")
print(f"Potential energy: {potential:.10e} J")
print(f"Total energy: {total:.10e} J")

print("Energy test passed successfully.")