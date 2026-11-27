import numpy as np
import matplotlib.pyplot as plt

from src.electrostatics import COULOMB_CONSTANT
from src.energy import calculate_particle_energy

from src.particle_dynamics import (
    simulate_euler,
    simulate_rk4,
    simulate_verlet
)


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

orbits = 100
steps_per_orbit = 1000

dt = period / steps_per_orbit
steps = orbits * steps_per_orbit

times = np.arange(steps + 1) * dt

methods = {
    "Euler": (simulate_euler, "orange"),
    "RK4": (simulate_rk4, "deepskyblue"),
    "Velocity Verlet": (simulate_verlet, "limegreen")
}

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(12, 8))

for name, (solver, color) in methods.items():

    states = solver(
        initial_state,
        dt,
        steps,
        particle_charge,
        particle_mass,
        charges,
        charge_positions
    )

    energies = np.array([
        calculate_particle_energy(
            state,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )[2]
        for state in states
    ])

    initial_energy = energies[0]

    relative_error = (
        np.abs(energies - initial_energy)
        / abs(initial_energy)
    )

    print(f"\n{name}")

    print(
        f"Maximum relative energy error: "
        f"{np.max(relative_error):.10e}"
    )

    print(
        f"Final relative energy error: "
        f"{relative_error[-1]:.10e}"
    )

    ax.semilogy(
        times / period,
        np.where(relative_error > 0, relative_error, np.nan),
        color=color,
        linewidth=1.5,
        label=name
    )

ax.set_xlabel("Orbital periods")
ax.set_ylabel("Relative energy error")

ax.set_title(
    "Long-Term Energy Conservation Analysis"
)

ax.grid(True, which="both", alpha=0.15)

ax.legend()

fig.tight_layout()

plt.show()