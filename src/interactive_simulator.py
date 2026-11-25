import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button
from src.grid import create_grid
from src.field_solver import calculate_field_grid_vectorized


def run_interactive_simulator():

    plt.style.use("dark_background")

    charges = np.array([5e-9, -5e-9])

    charge_positions = np.array([
        [-2.0, 0.0],
        [2.0, 0.0]
    ])

    initial_positions = charge_positions.copy()

    selected_charge = None

    X, Y = create_grid(-5, 5, -5, 5, 100)

    fig, ax = plt.subplots(figsize=(11, 9))

    fig.subplots_adjust(bottom=0.25)

    def update_field():

        ax.clear()

        Ex, Ey = calculate_field_grid_vectorized(
            charges,
            charge_positions,
            X,
            Y
        )

        magnitude = np.hypot(Ex, Ey)

        ax.streamplot(
            X, Y, Ex, Ey,
            color=np.log1p(magnitude),
            cmap="turbo",
            density=1.5,
            linewidth=1.0
        )

        for q, position in zip(charges, charge_positions):

            color = "red" if q > 0 else "dodgerblue" if q < 0 else "gray"

            ax.scatter(
                position[0],
                position[1],
                color=color,
                s=100,
                zorder=5
            )

            ax.annotate(
                f"{q * 1e9:+.2f} nC",
                position,
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                color=color
            )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Interactive Electric Field Simulator")

        fig.canvas.draw_idle()

    slider_ax1 = fig.add_axes([0.20, 0.15, 0.60, 0.03])
    slider_ax2 = fig.add_axes([0.20, 0.10, 0.60, 0.03])

    slider_q1 = Slider(
        slider_ax1,
        "Charge 1 (nC)",
        -10,
        10,
        valinit=5
    )

    slider_q2 = Slider(
        slider_ax2,
        "Charge 2 (nC)",
        -10,
        10,
        valinit=-5
    )

    def update_charges(value):

        charges[0] = slider_q1.val * 1e-9
        charges[1] = slider_q2.val * 1e-9

        update_field()

    slider_q1.on_changed(update_charges)
    slider_q2.on_changed(update_charges)

    def on_press(event):

        nonlocal selected_charge

        if event.inaxes != ax or event.button != 1:
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

    def on_motion(event):

        if selected_charge is None:
            return

        if event.inaxes != ax:
            return

        charge_positions[selected_charge] = [
            event.xdata,
            event.ydata
        ]

        update_field()

    def on_release(event):

        nonlocal selected_charge

        selected_charge = None

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

    reset_ax = fig.add_axes([0.42, 0.025, 0.16, 0.04])

    reset_button = Button(reset_ax, "Reset")

    def reset(event):

        nonlocal selected_charge

        selected_charge = None

        charge_positions[:] = initial_positions

        slider_q1.reset()
        slider_q2.reset()

        update_field()

    reset_button.on_clicked(reset)

    update_field()

    plt.show()


if __name__ == "__main__":
    run_interactive_simulator()