import numpy as np
import matplotlib.pyplot as plt


def plot_electric_field(X, Y, Ex, Ey, charges, charge_positions):

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
    
    for q, position in zip(charges, charge_positions):

        color = "red" if q > 0 else "dodgerblue"

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
    ax.set_title("Electric Field Visualization")
    fig.tight_layout()
    plt.show()