import numpy as np

from src.electrostatics import COULOMB_CONSTANT


def calculate_probe_values(charges, charge_positions, point):

    point = np.asarray(point, dtype=float)

    displacement = point - charge_positions

    distances = np.linalg.norm(displacement, axis=1)

    active = charges != 0

    displacement = displacement[active]
    distances = distances[active]
    charges = charges[active]

    if np.any(distances == 0):
        raise ValueError("Probe located at a point charge")

    field = COULOMB_CONSTANT * np.sum(
        charges[:, None] * displacement /
        distances[:, None]**3,
        axis=0
    )

    potential = COULOMB_CONSTANT * np.sum(
        charges / distances
    )

    magnitude = np.linalg.norm(field)

    return field, magnitude, potential