import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize
from src.grid import create_grid
from src.field_probe import calculate_probe_values
from src.field_solver import (
    calculate_field_grid_vectorized,
    calculate_potential_grid
)


def run_interactive_simulator():

    plt.style.use("dark_background")

    charges = np.array([5e-9, -5e-9])

    charge_positions = np.array([
        [-2.0, 0.0],
        [2.0, 0.0]
    ])

    initial_charges = charges.copy()
    initial_positions = charge_positions.copy()

    selected_charge = 0
    dragging = False
    slider_changing = False
    probe_point = None

    view_modes = [
        "Electric Field",
        "Equipotential",
        "Combined"
    ]

    view_index = 0

    last_update = 0.0
    update_interval = 1 / 15

    X_fast, Y_fast = create_grid(-5, 5, -5, 5, 60)
    X_full, Y_full = create_grid(-5, 5, -5, 5, 200)

    fig, ax = plt.subplots(figsize=(11, 9))

    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(bottom=0.27, right=0.72)

    probe_text = fig.text(
        0.74, 0.75,
        "Right-click to place probe",
        color="white",
        fontsize=10,
        va="top",
        linespacing=1.7
    )

    def draw_probe():

        if probe_point is None:

            probe_text.set_text(
                "Right-click to place probe"
            )

            return

        x, y = probe_point

        ax.scatter(
            x, y,
            marker="x",
            color="yellow",
            s=70,
            zorder=8
        )

        try:

            field, magnitude, potential = calculate_probe_values(
                charges,
                charge_positions,
                probe_point
            )

        except ValueError:

            probe_text.set_text(
                f"FIELD PROBE\n\n"
                f"x = {x:.3f} m\n"
                f"y = {y:.3f} m\n\n"
                "Undefined at point charge"
            )

            return

        probe_text.set_text(
            f"FIELD PROBE\n\n"
            f"x = {x:.3f} m\n"
            f"y = {y:.3f} m\n\n"
            f"Ex = {field[0]:.4g} N/C\n"
            f"Ey = {field[1]:.4g} N/C\n"
            f"|E| = {magnitude:.4g} N/C\n"
            f"V = {potential:.4g} V"
        )

        if np.isfinite(magnitude) and magnitude > 0:

            direction = 0.65 * field / magnitude

            ax.annotate(
                "",
                xy=probe_point + direction,
                xytext=probe_point,
                arrowprops=dict(
                    arrowstyle="->",
                    color="yellow",
                    lw=2
                ),
                zorder=9
            )

    def clear_probe(event):

        nonlocal probe_point

        probe_point = None

        update_field()

    def update_field(fast=False):

        nonlocal last_update

        current_time = time.perf_counter()

        if fast and current_time - last_update < update_interval:
            return

        last_update = current_time

        if fast:

            X, Y = X_fast, Y_fast
            density = 0.8
            linewidth = 0.8
            number_of_levels = 7

        else:

            X, Y = X_full, Y_full
            density = 1.5
            linewidth = 1.0
            number_of_levels = 13

        ax.clear()
        ax.set_facecolor("#0d1117")

        current_view = view_modes[view_index]

        if current_view in ["Electric Field", "Combined"]:

            Ex, Ey = calculate_field_grid_vectorized(
                charges,
                charge_positions,
                X,
                Y
            )

            magnitude = np.hypot(Ex, Ey)

            log_magnitude = np.log1p(magnitude)

            finite_values = log_magnitude[
                np.isfinite(log_magnitude)
            ]

            if finite_values.size and np.max(finite_values) > 0:

                upper_limit = np.percentile(
                    finite_values,
                    95
                )

                norm = Normalize(
                    vmin=0,
                    vmax=max(upper_limit, 1e-12),
                    clip=True
                )

                ax.streamplot(
                    X, Y, Ex, Ey,
                    color=log_magnitude,
                    cmap="turbo",
                    norm=norm,
                    density=density,
                    linewidth=linewidth
                )

        if current_view in ["Equipotential", "Combined"]:

            V = calculate_potential_grid(
                charges,
                charge_positions,
                X,
                Y
            )

            finite_values = V[np.isfinite(V)]

            if finite_values.size:

                limit = np.percentile(
                    np.abs(finite_values),
                    95
                )

                if limit > 0:

                    levels = np.linspace(
                        -limit,
                        limit,
                        number_of_levels
                    )

                    levels = levels[
                        (levels > np.min(finite_values)) &
                        (levels < np.max(finite_values))
                    ]

                    if levels.size:

                        contours = ax.contour(
                            X, Y, V,
                            levels=levels,
                            colors="#d1d5db",
                            linewidths=0.9,
                            linestyles="dashed",
                            alpha=0.8
                        )

                        if not fast:

                            ax.clabel(
                                contours,
                                fontsize=8,
                                fmt="%.1f V"
                            )

        for i, (q, position) in enumerate(
            zip(charges, charge_positions)
        ):

            color = (
                "#ff4757" if q > 0
                else "#38bdf8" if q < 0
                else "gray"
            )

            ax.scatter(
                position[0],
                position[1],
                color=color,
                s=100,
                edgecolors="white",
                zorder=5
            )

            if i == selected_charge:

                ax.scatter(
                    position[0],
                    position[1],
                    s=250,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=1.5,
                    zorder=6
                )

            ax.annotate(
                f"{q * 1e9:+.2f} nC",
                position,
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold"
            )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.set_title(
            f"{current_view} | {len(charges)} charges",
            fontsize=14,
            fontweight="bold"
        )

        ax.grid(alpha=0.08)

        draw_probe()

        fig.canvas.draw_idle()

    slider_ax = fig.add_axes(
        [0.25, 0.19, 0.55, 0.03]
    )

    charge_slider = Slider(
        slider_ax,
        "Selected charge (nC)",
        -10,
        10,
        valinit=5,
        valstep=0.1
    )

    def sync_slider():

        charge_slider.eventson = False

        if selected_charge is None:

            charge_slider.set_val(0)
            charge_slider.set_active(False)

        else:

            charge_slider.set_active(True)

            charge_slider.set_val(
                charges[selected_charge] * 1e9
            )

        charge_slider.eventson = True

    def update_charge(value):

        nonlocal slider_changing

        if selected_charge is None:
            return

        charges[selected_charge] = value * 1e-9

        slider_changing = True

        update_field(fast=True)

    charge_slider.on_changed(update_charge)

    def on_press(event):

        nonlocal selected_charge
        nonlocal dragging
        nonlocal probe_point

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button == 3:

            probe_point = np.array([
                event.xdata,
                event.ydata
            ])

            update_field()

            return

        if event.button != 1:
            return

        if len(charges) == 0:
            return

        mouse_position = np.array([
            event.xdata,
            event.ydata
        ])

        distances = np.linalg.norm(
            charge_positions - mouse_position,
            axis=1
        )

        nearest_charge = np.argmin(distances)

        if distances[nearest_charge] < 0.3:

            selected_charge = nearest_charge
            dragging = True

            sync_slider()
            update_field(fast=True)

    def on_motion(event):

        if not dragging or selected_charge is None:
            return

        if event.inaxes != ax:
            return

        charge_positions[selected_charge] = [
            event.xdata,
            event.ydata
        ]

        update_field(fast=True)

    def on_release(event):

        nonlocal dragging
        nonlocal slider_changing

        if dragging or slider_changing:

            dragging = False
            slider_changing = False

            update_field()

    def add_charge(value):

        nonlocal charges
        nonlocal charge_positions
        nonlocal selected_charge
        nonlocal dragging

        position = np.array([0.0, 0.0])

        for _ in range(100):

            if not len(charges):
                break

            distances = np.linalg.norm(
                charge_positions - position,
                axis=1
            )

            if np.all(distances > 0.4):
                break

            position = np.random.uniform(
                -3, 3, size=2
            )

        charges = np.append(charges, value)

        charge_positions = np.vstack([
            charge_positions,
            position
        ])

        selected_charge = len(charges) - 1
        dragging = False

        sync_slider()
        update_field()

    def add_positive(event):

        add_charge(5e-9)

    def add_negative(event):

        add_charge(-5e-9)

    def remove_charge(event):

        nonlocal charges
        nonlocal charge_positions
        nonlocal selected_charge
        nonlocal dragging

        if selected_charge is None:
            return

        charges = np.delete(
            charges,
            selected_charge
        )

        charge_positions = np.delete(
            charge_positions,
            selected_charge,
            axis=0
        )

        selected_charge = 0 if len(charges) else None
        dragging = False

        sync_slider()
        update_field()

    def change_view(event):

        nonlocal view_index

        view_index = (
            view_index + 1
        ) % len(view_modes)

        view_button.label.set_text(
            f"View: {view_modes[view_index]}"
        )

        update_field()

    def reset(event):

        nonlocal charges
        nonlocal charge_positions
        nonlocal selected_charge
        nonlocal dragging
        nonlocal slider_changing
        nonlocal probe_point

        probe_point = None

        charges = initial_charges.copy()
        charge_positions = initial_positions.copy()

        selected_charge = 0
        dragging = False
        slider_changing = False

        sync_slider()
        update_field()

    positive_ax = fig.add_axes(
        [0.10, 0.105, 0.19, 0.045]
    )

    negative_ax = fig.add_axes(
        [0.30, 0.105, 0.19, 0.045]
    )

    remove_ax = fig.add_axes(
        [0.50, 0.105, 0.19, 0.045]
    )

    reset_ax = fig.add_axes(
        [0.70, 0.105, 0.19, 0.045]
    )

    view_ax = fig.add_axes(
        [0.20, 0.035, 0.39, 0.045]
    )

    clear_probe_ax = fig.add_axes(
        [0.62, 0.035, 0.19, 0.045]
    )

    positive_button = Button(
        positive_ax,
        "+ Positive",
        color="#54242b",
        hovercolor="#a83242"
    )

    negative_button = Button(
        negative_ax,
        "+ Negative",
        color="#163d55",
        hovercolor="#2563a0"
    )

    remove_button = Button(
        remove_ax,
        "Remove",
        color="#30363d",
        hovercolor="#555555"
    )

    reset_button = Button(
        reset_ax,
        "Reset",
        color="#30363d",
        hovercolor="#555555"
    )

    view_button = Button(
        view_ax,
        "View: Electric Field",
        color="#34305c",
        hovercolor="#51498a"
    )

    clear_probe_button = Button(
        clear_probe_ax,
        "Clear Probe",
        color="#30363d",
        hovercolor="#555555"
    )

    positive_button.on_clicked(add_positive)
    negative_button.on_clicked(add_negative)
    remove_button.on_clicked(remove_charge)
    reset_button.on_clicked(reset)
    view_button.on_clicked(change_view)
    clear_probe_button.on_clicked(clear_probe)

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release
    )

    update_field()

    plt.show()


if __name__ == "__main__":
    run_interactive_simulator()