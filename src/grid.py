import numpy as np
import numpy as np


def create_grid(x_min, x_max, y_min, y_max, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    X, Y = np.meshgrid(x, y)

    return X, Y