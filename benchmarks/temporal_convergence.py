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

initial_state = np.array([
    radius,
    0.0,
    0.0,
    speed
])

step_counts = [100, 200, 400, 800, 1600, 3200]

time_steps = []
euler_errors = []
rk4_errors = []

for steps in step_counts:

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

    theta = np.linspace(
        0, 2 * np.pi, steps + 1
    )

    exact_positions = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])

    euler_error = np.max(
        np.linalg.norm(
            euler_states[:, :2] - exact_positions,
            axis=1
        )
    )

    rk4_error = np.max(
        np.linalg.norm(
            rk4_states[:, :2] - exact_positions,
            axis=1
        )
    )

    time_steps.append(dt)
    euler_errors.append(euler_error)
    rk4_errors.append(rk4_error)

    print(f"Steps: {steps}")
    print(f"Time step: {dt:.8f} s")
    print(f"Euler error: {euler_error:.10e} m")
    print(f"RK4 error: {rk4_error:.10e} m")
    print()

euler_order = np.polyfit(
    np.log(time_steps),
    np.log(euler_errors),
    1
)[0]

rk4_order = np.polyfit(
    np.log(time_steps),
    np.log(rk4_errors),
    1
)[0]

print(f"Estimated Euler order: {euler_order:.4f}")
print(f"Estimated RK4 order: {rk4_order:.4f}")

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 7))

ax.loglog(
    time_steps,
    euler_errors,
    "o-",
    color="orange",
    linewidth=2,
    label=f"Euler (order {euler_order:.2f})"
)

ax.loglog(
    time_steps,
    rk4_errors,
    "s-",
    color="deepskyblue",
    linewidth=2,
    label=f"RK4 (order {rk4_order:.2f})"
)

ax.set_xlabel("Time step (s)")
ax.set_ylabel("Maximum position error (m)")
ax.set_title("Temporal Convergence Analysis")

ax.grid(True, which="both", alpha=0.15)
ax.legend()

fig.tight_layout()

plt.show()