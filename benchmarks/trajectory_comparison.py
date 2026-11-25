import numpy as np
import matplotlib.pyplot as plt

from src.electrostatics import COULOMB_CONSTANT
from src.particle_dynamics import simulate_euler, simulate_rk4

charges = np.array([-5e-9])
charge_positions = np.array([[0.0, 0.0]])

particle_charge = 1e-9
particle_mass = 1e-9

radius = 2.0

speed = np.sqrt(
    COULOMB_CONSTANT
    * abs(charges[0] * particle_charge)
    / (particle_mass * radius)
)

period = 2 * np.pi * radius / speed

steps = 1000
dt = period / steps

initial_state = np.array([
    radius,
    0.0,
    0.0,
    speed
])

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

theta = np.linspace(0, 2 * np.pi, steps + 1)

exact_x = radius * np.cos(theta)
exact_y = radius * np.sin(theta)

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(
    euler_states[:, 0],
    euler_states[:, 1],
    color="orange",
    linewidth=2,
    label="Euler"
)

ax.plot(
    rk4_states[:, 0],
    rk4_states[:, 1],
    color="deepskyblue",
    linewidth=3,
    label="RK4"
)

ax.plot(
    exact_x,
    exact_y,
    color="white",
    linestyle="--",
    linewidth=1.5,
    label="Exact solution"
)

ax.scatter(
    0, 0,
    color="dodgerblue",
    s=100,
    label="Central charge"
)

ax.scatter(
    radius, 0,
    color="lime",
    s=60,
    label="Initial position"
)

ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Orbital Trajectory Comparison")
ax.legend()
ax.grid(alpha=0.15)

fig.tight_layout()

plt.show()
