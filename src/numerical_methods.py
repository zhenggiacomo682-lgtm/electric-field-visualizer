import numpy as np


def calculate_field_from_potential(V, X, Y):

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    dV_dy, dV_dx = np.gradient(
        V, dy, dx, edge_order=2
    )

    Ex = -dV_dx
    Ey = -dV_dy

    return Ex, Ey