import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize

from src.grid import create_grid
from src.field_probe import calculate_probe_values
from src.particle_simulator import ParticleSimulator
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

    particle = None
    placing_particle = False
    particle_marker = None
    trajectory_line = None

    particle_status = "Place a particle to begin"

    view_modes = [
        "Electric Field",
        "Equipotential",
        "Combined"
    ]

    methods = [
        "Euler",
        "RK4",
        "Verlet"
    ]

    view_index = 0
    method_index = 1

    last_update = 0.0
    update_interval = 1 / 15

    X_fast, Y_fast = create_grid(-5, 5, -5, 5, 60)
    X_full, Y_full = create_grid(-5, 5, -5, 5, 200)

    fig, ax = plt.subplots(figsize=(13, 9))

    fig.patch.set_facecolor("#0d1117")

    fig.subplots_adjust(
        left=0.08,
        right=0.70,
        bottom=0.34,
        top=0.94
    )

    probe_text = fig.text(
        0.74,
        0.88,
        "Right-click to place probe",
        color="white",
        fontsize=10,
        va="top",
        linespacing=1.5
    )

    particle_text = fig.text(
        0.74,
        0.59,
        particle_status,
        color="white",
        fontsize=9,
        va="top",
        linespacing=1.2
    )

    def refresh_particle():

        if particle_marker is None or trajectory_line is None:
            return

        if particle is None:

            particle_marker.set_data([], [])
            trajectory_line.set_data([], [])
            particle_text.set_text(particle_status)

            return

        position = particle.get_position()
        velocity = particle.get_velocity()
        trajectory = particle.get_trajectory()

        particle_marker.set_data(
            [position[0]],
            [position[1]]
        )

        trajectory_line.set_data(
            trajectory[:, 0],
            trajectory[:, 1]
        )

        try:

            kinetic, potential, total = particle.get_energy(
                charges,
                charge_positions
            )

            energy_text = (
                f"K = {kinetic:.3e} J\n"
                f"U = {potential:.3e} J\n"
                f"E = {total:.3e} J"
            )

        except ValueError:

            energy_text = "Energy undefined"

        particle_text.set_text(
            f"PARTICLE | {particle.method}\n\n"
            f"t = {particle.time:.3f} s\n"
            f"x = {position[0]:.3f} m\n"
            f"y = {position[1]:.3f} m\n"
            f"vx = {velocity[0]:.3f} m/s\n"
            f"vy = {velocity[1]:.3f} m/s\n\n"
            f"{energy_text}\n\n"
            f"{particle_status}"
        )

    def pause_particle(status="Paused"):

        nonlocal particle_status

        if particle is not None:
            particle.pause()

        particle_status = status

        start_button.label.set_text("Start")

    def place_particle(event):

        nonlocal placing_particle

        pause_particle("Click the field to place particle")

        placing_particle = True

        refresh_particle()
        fig.canvas.draw_idle()

    def toggle_particle(event):

        nonlocal particle_status
        nonlocal placing_particle

        if particle is None:

            place_particle(event)

            return

        placing_particle = False

        if particle.running:

            pause_particle()

        else:

            particle.start()

            particle_status = "Running"

            start_button.label.set_text("Pause")

        refresh_particle()
        fig.canvas.draw_idle()

    def reset_particle(event):

        nonlocal placing_particle

        placing_particle = False

        if particle is not None:

            particle.reset()
            pause_particle("Ready")

        else:

            pause_particle("Place a particle to begin")

        refresh_particle()
        fig.canvas.draw_idle()

    def change_integrator(event):

        nonlocal method_index

        method_index = (
            method_index + 1
        ) % len(methods)

        method = methods[method_index]

        if particle is not None:

            pause_particle("Integrator changed")

            particle.method = method

        integrator_button.label.set_text(
            f"Integrator: {method}"
        )

        refresh_particle()
        fig.canvas.draw_idle()

    def animate_particle():

        if particle is None or not particle.running:
            return

        for _ in range(10):

            position = particle.get_position()

            active_positions = charge_positions[
                charges != 0
            ]

            if not np.all(np.isfinite(particle.state)):

                pause_particle("Invalid numerical state")

                break

            if np.any(np.abs(position) > 5):

                pause_particle("Outside simulation domain")

                break

            if np.any(
                np.linalg.norm(
                    active_positions - position,
                    axis=1
                ) < 0.08
            ):

                pause_particle("Near point charge: stopped")

                break

            previous_state = particle.state.copy()
            previous_time = particle.time
            previous_length = len(particle.trajectory)

            try:

                particle.step(
                    charges,
                    charge_positions
                )

                if not np.all(np.isfinite(particle.state)):

                    raise FloatingPointError(
                        "Non-finite particle state"
                    )

            except (ValueError, FloatingPointError):

                particle.state = previous_state
                particle.time = previous_time

                del particle.trajectory[previous_length:]

                pause_particle("Numerical singularity: stopped")

                break

        refresh_particle()
        fig.canvas.draw_idle()

    def draw_probe():

        if probe_point is None:

            probe_text.set_text(
                "Right-click to place probe"
            )

            return

        x, y = probe_point

        ax.scatter(
            x,
            y,
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
                "FIELD PROBE\n\n"
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
        nonlocal particle_marker
        nonlocal trajectory_line

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

        active = charges != 0

        active_charges = charges[active]
        active_positions = charge_positions[active]

        if current_view in ["Electric Field", "Combined"]:

            if len(active_charges) > 0:

                Ex, Ey = calculate_field_grid_vectorized(
                    active_charges,
                    active_positions,
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
                        X,
                        Y,
                        Ex,
                        Ey,
                        color=log_magnitude,
                        cmap="turbo",
                        norm=norm,
                        density=density,
                        linewidth=linewidth
                    )

        if current_view in ["Equipotential", "Combined"]:

            if len(active_charges) > 0:

                V = calculate_potential_grid(
                    active_charges,
                    active_positions,
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
                                X,
                                Y,
                                V,
                                levels=levels,
                                colors="white",
                                linewidths=0.8,
                                linestyles="dashed",
                                alpha=0.6
                            )

                            if not fast:

                                ax.clabel(
                                    contours,
                                    inline=True,
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

        trajectory_line, = ax.plot(
            [],
            [],
            color="#a3e635",
            lw=1.5,
            zorder=7
        )

        particle_marker, = ax.plot(
            [],
            [],
            "o",
            color="#a3e635",
            markersize=7,
            markeredgecolor="white",
            zorder=10
        )

        refresh_particle()

        fig.canvas.draw_idle()

    slider_ax = fig.add_axes(
        [0.25, 0.26, 0.55, 0.03]
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

        pause_particle("Source charge changed")

        charges[selected_charge] = value * 1e-9

        slider_changing = True

        update_field(fast=True)

    def on_press(event):

        nonlocal selected_charge
        nonlocal dragging
        nonlocal probe_point
        nonlocal particle
        nonlocal placing_particle
        nonlocal particle_status

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        position = np.array([
            event.xdata,
            event.ydata
        ])

        if event.button == 3:

            probe_point = position

            update_field()

            return

        if event.button != 1:
            return

        if placing_particle:

            active_positions = charge_positions[
                charges != 0
            ]

            if np.any(
                np.linalg.norm(
                    active_positions - position,
                    axis=1
                ) < 0.08
            ):

                particle_status = "Choose a point away from charges"

                refresh_particle()
                fig.canvas.draw_idle()

                return

            particle = ParticleSimulator(
                position=position,
                velocity=[0.0, 1.0],
                particle_charge=1e-9,
                particle_mass=1e-9,
                dt=0.001,
                method=methods[method_index]
            )

            placing_particle = False

            pause_particle("Ready")

            refresh_particle()

            fig.canvas.draw_idle()

            return

        if len(charges) == 0:
            return

        distances = np.linalg.norm(
            charge_positions - position,
            axis=1
        )

        nearest_charge = np.argmin(distances)

        if distances[nearest_charge] < 0.3:

            pause_particle("Source charge selected")

            selected_charge = nearest_charge

            dragging = True

            sync_slider()

            update_field(fast=True)

    def on_motion(event):

        if not dragging or selected_charge is None:
            return

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
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
                -3,
                3,
                size=2
            )

        pause_particle("Source charge added")

        charges = np.append(
            charges,
            value
        )

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

        pause_particle("Source charge removed")

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
        nonlocal particle
        nonlocal placing_particle

        pause_particle("Place a particle to begin")

        probe_point = None
        particle = None
        placing_particle = False

        charges = initial_charges.copy()

        charge_positions = initial_positions.copy()

        selected_charge = 0

        dragging = False
        slider_changing = False

        sync_slider()

        update_field()

    def create_button(
        bounds,
        label,
        callback,
        color="#30363d",
        hovercolor="#555555"
    ):

        button_ax = fig.add_axes(bounds)

        button = Button(
            button_ax,
            label,
            color=color,
            hovercolor=hovercolor
        )

        button.on_clicked(callback)

        return button

    positive_button = create_button(
        [0.10, 0.18, 0.19, 0.045],
        "+ Positive",
        add_positive,
        "#54242b",
        "#a83242"
    )

    negative_button = create_button(
        [0.30, 0.18, 0.19, 0.045],
        "+ Negative",
        add_negative,
        "#163d55",
        "#2563a0"
    )

    remove_button = create_button(
        [0.50, 0.18, 0.19, 0.045],
        "Remove",
        remove_charge
    )

    reset_button = create_button(
        [0.70, 0.18, 0.19, 0.045],
        "Reset",
        reset
    )

    view_button = create_button(
        [0.20, 0.115, 0.39, 0.045],
        "View: Electric Field",
        change_view,
        "#34305c",
        "#51498a"
    )

    clear_probe_button = create_button(
        [0.62, 0.115, 0.19, 0.045],
        "Clear Probe",
        clear_probe
    )

    place_button = create_button(
        [0.10, 0.04, 0.19, 0.045],
        "Place Particle",
        place_particle
    )

    start_button = create_button(
        [0.30, 0.04, 0.19, 0.045],
        "Start",
        toggle_particle,
        "#24543a",
        "#387c55"
    )

    particle_reset_button = create_button(
        [0.50, 0.04, 0.19, 0.045],
        "Reset Particle",
        reset_particle
    )

    integrator_button = create_button(
        [0.70, 0.04, 0.19, 0.045],
        "Integrator: RK4",
        change_integrator,
        "#34305c",
        "#51498a"
    )

    charge_slider.on_changed(update_charge)

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

    timer = fig.canvas.new_timer(
        interval=30
    )

    timer.add_callback(animate_particle)

    fig.canvas.mpl_connect(
        "close_event",
        lambda event: timer.stop()
    )

    update_field()

    timer.start()

    plt.show()


if __name__ == "__main__":
    run_interactive_simulator()