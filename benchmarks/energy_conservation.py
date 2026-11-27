import numpy as np
import matplotlib.pyplot as plt

from src.electrostatics import COULOMB_CONSTANT
from src.energy import calculate_particle_energy
from src.particle_dynamics import simulate_euler, simulate_rk4


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

orbits = 10
steps_per_orbit = 1000

dt = period / steps_per_orbit
steps = orbits * steps_per_orbit

times = np.arange(steps + 1) * dt

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


def calculate_energy_history(states):

    return np.array([
        calculate_particle_energy(
            state,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )[2]
        for state in states
    ])


euler_energy = calculate_energy_history(euler_states)
rk4_energy = calculate_energy_history(rk4_states)

initial_energy = euler_energy[0]

euler_error = (
    np.abs(euler_energy - initial_energy)
    / abs(initial_energy)
)

rk4_error = (
    np.abs(rk4_energy - initial_energy)
    / abs(initial_energy)
)

print(f"Initial energy: {initial_energy:.10e} J")
print(f"Euler maximum relative error: {np.max(euler_error):.10e}")
print(f"RK4 maximum relative error: {np.max(rk4_error):.10e}")

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(11, 7))

ax.semilogy(
    times[1:] / period,
    euler_error[1:],
    color="orange",
    linewidth=2,
    label="Euler"
)

ax.semilogy(
    times[1:] / period,
    rk4_error[1:],
    color="deepskyblue",
    linewidth=2,
    label="RK4"
)

ax.set_xlabel("Orbital periods")
ax.set_ylabel("Relative energy error")
ax.set_title("Energy Conservation Analysis")

ax.grid(True, which="both", alpha=0.15)
ax.legend()

fig.tight_layout()

plt.show()