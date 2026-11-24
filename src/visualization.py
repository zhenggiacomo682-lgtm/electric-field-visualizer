import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def plot_electric_field(X, Y, Ex, Ey, V, charges, charge_positions):

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(11, 9))

    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    magnitude = np.hypot(Ex, Ey)

    log_magnitude = np.log1p(magnitude)

    finite_values = log_magnitude[np.isfinite(log_magnitude)]

    upper_limit = np.percentile(finite_values, 95) if finite_values.size else 1.0

    norm = Normalize(
        vmin=0,
        vmax=max(upper_limit, 1e-12),
        clip=True
    )

    stream = ax.streamplot(
        X, Y, Ex, Ey,
        color=log_magnitude,
        cmap="turbo",
        norm=norm,
        density=1.5,
        linewidth=1.1,
        arrowsize=1.0
    )

    potential_limit = np.nanpercentile(np.abs(V), 95)

    if np.isfinite(potential_limit) and potential_limit > 0:

        levels = np.linspace(
            -potential_limit,
            potential_limit,
            11
        )

        contours = ax.contour(
            X, Y, V,
            levels=levels,
            colors="#d1d5db",
            linewidths=0.8,
            linestyles="dashed",
            alpha=0.65
        )

        ax.clabel(
            contours,
            fontsize=8,
            fmt="%.1f V"
        )

    for q, position in zip(charges, charge_positions):

        color = "#ff4757" if q > 0 else "#38bdf8" if q < 0 else "gray"

        symbol = "+" if q > 0 else "−" if q < 0 else "0"

        ax.scatter(
            position[0],
            position[1],
            color=color,
            s=100,
            edgecolors="white",
            linewidths=0.8,
            zorder=5
        )

        ax.text(
            position[0],
            position[1],
            symbol,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6
        )

        ax.annotate(
            f"{q * 1e9:+.2g} nC",
            position,
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=10,
            fontweight="bold"
        )

    legend_elements = [
        Line2D(
            [0], [0],
            color="#38bdf8",
            linewidth=2,
            label="Electric field lines"
        ),
        Line2D(
            [0], [0],
            color="#d1d5db",
            linestyle="--",
            label="Equipotential lines"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        facecolor="#161b22",
        framealpha=0.9
    )

    cbar = fig.colorbar(
        stream.lines,
        ax=ax,
        pad=0.03,
        shrink=0.85,
        extend="max"
    )

    cbar.set_label("log(1 + |E| / (1 N/C))")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    ax.set_title(
        "Electric Field & Equipotential Lines",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    ax.grid(alpha=0.08)

    fig.tight_layout()

  


def plot_electric_potential(X, Y, V, charges, charge_positions):

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    finite_values = np.abs(V[np.isfinite(V)])

    limit = np.percentile(finite_values, 95) if finite_values.size else 1.0

    limit = max(limit, 1e-12)

    levels = np.linspace(-limit, limit, 101)

    heatmap = ax.contourf(
        X, Y, V,
        levels=levels,
        cmap="RdBu_r",
        extend="both"
    )

    if finite_values.size and np.max(finite_values) > 0:

        contours = ax.contour(
            X, Y, V,
            levels=np.linspace(-limit, limit, 11),
            colors="black",
            linewidths=0.6,
            alpha=0.5
        )

        ax.clabel(
            contours,
            fontsize=8,
            fmt="%.1f V"
        )

    for q, position in zip(charges, charge_positions):

        color = "#ff4757" if q > 0 else "#38bdf8" if q < 0 else "gray"

        ax.scatter(
            position[0],
            position[1],
            color=color,
            s=100,
            edgecolors="white",
            zorder=5
        )

        ax.annotate(
            f"{q * 1e9:+.2g} nC",
            position,
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            color="black",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="none"
            )
        )

    cbar = fig.colorbar(heatmap, ax=ax)

    cbar.set_label("Electric potential (V)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.set_title(
        "Electric Potential Heatmap",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    fig.tight_layout()
    
    
def plot_potential_3d(X, Y, V, charges, charge_positions):

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(11, 8))

    fig.patch.set_facecolor("#0d1117")

    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("#0d1117")

    finite_values = np.abs(V[np.isfinite(V)])

    limit = np.percentile(finite_values, 98) if finite_values.size else 1.0

    limit = max(limit, 1e-12)

    V_display = np.clip(V, -limit, limit)

    norm = Normalize(vmin=-limit, vmax=limit)

    surface = ax.plot_surface(
        X, Y, V_display,
        cmap="RdBu_r",
        norm=norm,
        rcount=200,
        ccount=200,
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    for q, position in zip(charges, charge_positions):

        color = "#ff4757" if q > 0 else "#38bdf8" if q < 0 else "gray"

        ax.scatter(
            position[0],
            position[1],
            0,
            color=color,
            s=80
        )

    cbar = fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        pad=0.1
    )

    cbar.set_label("Electric potential (V)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Potential (V)")

    ax.set_title(
        "3D Electric Potential Surface",
        fontsize=17,
        fontweight="bold",
        pad=20
    )

    ax.view_init(elev=30, azim=-60)

    fig.tight_layout()
