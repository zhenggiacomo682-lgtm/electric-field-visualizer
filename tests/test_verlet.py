import numpy as np

from src.electrostatics import COULOMB_CONSTANT
from src.particle_dynamics import simulate_euler, simulate_verlet
from src.energy import calculate_particle_energy


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

initial_state = np.array([
    radius,
    0.0,
    0.0,
    velocity
])

period = 2 * np.pi * radius / velocity

steps = 1000
dt = period / steps

euler_states = simulate_euler(
    initial_state,
    dt,
    steps,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
)

verlet_states = simulate_verlet(
    initial_state,
    dt,
    steps,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
)

euler_error = np.linalg.norm(
    euler_states[-1, :2] - initial_state[:2]
)

verlet_error = np.linalg.norm(
    verlet_states[-1, :2] - initial_state[:2]
)

energies = np.array([
    calculate_particle_energy(
        state,
        particle_charge,
        particle_mass,
        charges,
        charge_positions
    )[2]
    for state in verlet_states
])

energy_error = np.max(
    np.abs(energies - energies[0]) / abs(energies[0])
)

assert verlet_states.shape == (steps + 1, 4)
assert np.all(np.isfinite(verlet_states))
assert verlet_error < euler_error
assert verlet_error < 1e-3
assert energy_error < 1e-6

print(f"Orbital period: {period:.6f} s")
print(f"Euler position error: {euler_error:.10e} m")
print(f"Verlet position error: {verlet_error:.10e} m")
print(f"Maximum relative energy error: {energy_error:.10e}")

print("Velocity Verlet test passed successfully.")
