import numpy as np
from src.electrostatics import total_electric_field


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