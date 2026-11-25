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

    initial_charges = charges.copy()
    initial_positions = charge_positions.copy()

    selected_charge = 0
    dragging = False

    X, Y = create_grid(-5, 5, -5, 5, 100)

    fig, ax = plt.subplots(figsize=(11, 9))

    fig.patch.set_facecolor("#0d1117")

    fig.subplots_adjust(bottom=0.22)

    def update_field():

        ax.clear()
        ax.set_facecolor("#0d1117")

        Ex, Ey = calculate_field_grid_vectorized(
            charges,
            charge_positions,
            X,
            Y
        )

        magnitude = np.hypot(Ex, Ey)

        if np.any(np.isfinite(magnitude) & (magnitude > 0)):

            ax.streamplot(
                X, Y, Ex, Ey,
                color=np.log1p(magnitude),
                cmap="turbo",
                density=1.5,
                linewidth=1.0
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
            f"Interactive Electric Field Simulator | {len(charges)} charges",
            fontsize=14,
            fontweight="bold"
        )

        ax.grid(alpha=0.08)

        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.25, 0.13, 0.55, 0.03])

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

        if selected_charge is None:
            return

        charges[selected_charge] = value * 1e-9

        update_field()

    charge_slider.on_changed(update_charge)

    def on_press(event):

        nonlocal selected_charge, dragging

        if event.inaxes != ax or event.button != 1:
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
            update_field()

    def on_motion(event):

        if not dragging or selected_charge is None:
            return

        if event.inaxes != ax:
            return

        charge_positions[selected_charge] = [
            event.xdata,
            event.ydata
        ]

        update_field()

    def on_release(event):

        nonlocal dragging

        dragging = False

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

            position = np.random.uniform(-3, 3, size=2)

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

    def reset(event):

        nonlocal charges
        nonlocal charge_positions
        nonlocal selected_charge
        nonlocal dragging

        charges = initial_charges.copy()
        charge_positions = initial_positions.copy()

        selected_charge = 0
        dragging = False

        sync_slider()
        update_field()

    positive_ax = fig.add_axes([0.10, 0.045, 0.19, 0.045])
    negative_ax = fig.add_axes([0.30, 0.045, 0.19, 0.045])
    remove_ax = fig.add_axes([0.50, 0.045, 0.19, 0.045])
    reset_ax = fig.add_axes([0.70, 0.045, 0.19, 0.045])

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

    positive_button.on_clicked(add_positive)
    negative_button.on_clicked(add_negative)
    remove_button.on_clicked(remove_charge)
    reset_button.on_clicked(reset)

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