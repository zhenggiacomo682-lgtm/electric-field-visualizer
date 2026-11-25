import numpy as np

from src.electrostatics import COULOMB_CONSTANT
from src.particle_dynamics import simulate_euler, simulate_rk4

charges = np.array([-5e-9])

charge_positions = np.array([
    [0.0, 0.0]
])

particle_charge = 1e-9
particle_mass = 1e-9

radius = 2.0

velocity = np.sqrt(
    COULOMB_CONSTANT
    * abs(charges[0] * particle_charge)
    / (particle_mass * radius)
)

period = 2 * np.pi * radius / velocity

initial_state = np.array([
    radius,
    0.0,
    0.0,
    velocity
])

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

rk4_states = simulate_rk4(
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

rk4_error = np.linalg.norm(
    rk4_states[-1, :2] - initial_state[:2]
)

print(f"Orbital period: {period:.6f} s")
print(f"Time step: {dt:.6f} s")
print(f"Euler position error: {euler_error:.10e} m")
print(f"RK4 position error: {rk4_error:.10e} m")
print(f"Accuracy improvement: {euler_error / rk4_error:.2f}x")

assert rk4_error < euler_error
assert rk4_error < 1e-6

print("RK4 test passed successfully.")