import numpy as np
import matplotlib.pyplot as plt


def plot_electric_field(X, Y, Ex, Ey, V, charges, charge_positions):

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 8))

    magnitude = np.sqrt(Ex**2 + Ey**2)

    stream = ax.streamplot(
        X, Y, Ex, Ey,
        color=np.log1p(magnitude),
        cmap="plasma",
        density=1.8,
        linewidth=0.8,
        arrowsize=0.9
    )

    potential_limit = np.nanpercentile(np.abs(V), 95)

    if potential_limit > 0:

        levels = np.linspace(
            -potential_limit,
            potential_limit,
            21
        )

        contours = ax.contour(
            X, Y, V,
            levels=levels,
            colors="white",
            linewidths=0.6,
            linestyles="dashed",
            alpha=0.6
        )

        ax.clabel(contours, fontsize=7, fmt="%.1f V")

    for q, position in zip(charges, charge_positions):

        color = "red" if q > 0 else "dodgerblue" if q < 0 else "gray"

        ax.scatter(
            position[0],
            position[1],
            color=color,
            s=40,
            zorder=5
        )

    cbar = fig.colorbar(stream.lines, ax=ax)

    cbar.set_label("log(1 + |E| / (1 N/C))")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Electric Field and Equipotential Lines")

    fig.tight_layout()

    plt.show()