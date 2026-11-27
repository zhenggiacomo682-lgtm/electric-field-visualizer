import numpy as np

from src.field_probe import calculate_probe_values


def particle_derivative(
    state,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    if particle_mass <= 0:
        raise ValueError("Particle mass must be positive")

    position = np.asarray(state[:2], dtype=float)
    velocity = np.asarray(state[2:], dtype=float)

    field, _, _ = calculate_probe_values(
        charges,
        charge_positions,
        position
    )

    acceleration = (
        particle_charge / particle_mass
    ) * field

    return np.concatenate([
        velocity,
        acceleration
    ])
    
def euler_step(
    state,
    dt,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    derivative = particle_derivative(
        state,
        particle_charge,
        particle_mass,
        charges,
        charge_positions
    )

    return state + dt * derivative
def simulate_euler(
    initial_state,
    dt,
    steps,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    if dt <= 0:
        raise ValueError("Time step must be positive")

    if steps < 1:
        raise ValueError("Number of steps must be positive")

    states = np.zeros((steps + 1, 4))

    states[0] = initial_state

    for i in range(steps):

        states[i + 1] = euler_step(
            states[i],
            dt,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )

    return states
def rk4_step(
    state,
    dt,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    def f(s):

        return particle_derivative(
            s,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )

    k1 = f(state)

    k2 = f(state + dt * k1 / 2)

    k3 = f(state + dt * k2 / 2)

    k4 = f(state + dt * k3)

    return state + dt / 6 * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


def simulate_rk4(
    initial_state,
    dt,
    steps,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    if dt <= 0:
        raise ValueError("Time step must be positive")

    if steps < 1:
        raise ValueError("Number of steps must be positive")

    states = np.zeros((steps + 1, 4))

    states[0] = initial_state

    for i in range(steps):

        states[i + 1] = rk4_step(
            states[i],
            dt,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )

    return states


def verlet_step(
    state,
    dt,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    position = np.asarray(state[:2], dtype=float)
    velocity = np.asarray(state[2:], dtype=float)

    derivative = particle_derivative(
        state,
        particle_charge,
        particle_mass,
        charges,
        charge_positions
    )

    acceleration = derivative[2:]

    new_position = (
        position
        + velocity * dt
        + 0.5 * acceleration * dt**2
    )

    field, _, _ = calculate_probe_values(
        charges,
        charge_positions,
        new_position
    )

    new_acceleration = (
        particle_charge / particle_mass
    ) * field

    new_velocity = (
        velocity
        + 0.5 * (acceleration + new_acceleration) * dt
    )

    return np.concatenate([
        new_position,
        new_velocity
    ])


def simulate_verlet(
    initial_state,
    dt,
    steps,
    particle_charge,
    particle_mass,
    charges,
    charge_positions
):

    if dt <= 0:
        raise ValueError("Time step must be positive")

    if steps < 1:
        raise ValueError("Number of steps must be positive")

    states = np.zeros((steps + 1, 4))

    states[0] = initial_state

    for i in range(steps):

        states[i + 1] = verlet_step(
            states[i],
            dt,
            particle_charge,
            particle_mass,
            charges,
            charge_positions
        )

    return states