import numpy as np

from src.field_probe import calculate_probe_values


def calculate_particle_energy(
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

    _, _, potential = calculate_probe_values(
        charges,
        charge_positions,
        position
    )

    kinetic_energy = (
        0.5 * particle_mass * np.dot(velocity, velocity)
    )

    potential_energy = particle_charge * potential

    total_energy = kinetic_energy + potential_energy

    return kinetic_energy, potential_energy, total_energy