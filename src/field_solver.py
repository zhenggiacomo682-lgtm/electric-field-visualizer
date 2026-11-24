import numpy as np
from src.electrostatics import total_electric_field, COULOMB_CONSTANT


def calculate_field_grid(charges, charge_positions, X, Y):

    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            observation_point = np.array([X[i, j], Y[i, j]])

            try:
                field = total_electric_field(
                    charges,
                    charge_positions,
                    observation_point
                )

                Ex[i, j] = field[0]
                Ey[i, j] = field[1]

            except ValueError:
                Ex[i, j] = np.nan
                Ey[i, j] = np.nan

    return Ex, Ey


def calculate_field_grid_vectorized(charges, charge_positions, X, Y):

    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)

    for q, position in zip(charges, charge_positions):

        dx = X - position[0]
        dy = Y - position[1]

        distance = np.sqrt(dx**2 + dy**2)

        distance[distance == 0] = np.nan

        Ex += COULOMB_CONSTANT * q * dx / distance**3
        Ey += COULOMB_CONSTANT * q * dy / distance**3

    return Ex, Ey