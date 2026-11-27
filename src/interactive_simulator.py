import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, TextBox

from src.domain_manager import DomainManager
from src.field_renderer import FieldRenderer
from src.dynamic_renderer import DynamicRenderer
from src.animation import AnimationEngine
from src.field_probe import calculate_probe_values
from src.particle_simulator import ParticleSimulator


def run_interactive_simulator():

    plt.style.use("dark_background")

    charges = np.array([5e-9, -5e-9])

    positions = np.array([
        [-2.0, 0.0],
        [2.0, 0.0]
    ])

    selected_charge = 0

    particle = None
    particle_marker = None
    trajectory_line = None

    probe_point = None

    dragging = False
    slider_changing = False
    preview_active = False
    preview_dirty = False
    field_dirty = False
    placing_particle = False
    syncing_slider = False

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

    last_preview_update = 0.0
    preview_interval = 1 / 25

    integration_time = 0.0

    domain = DomainManager()

    fig, ax = plt.subplots(
        figsize=(15, 10)
    )

    fig.patch.set_facecolor("#0d1117")

    fig.subplots_adjust(
        left=0.07,
        right=0.65,
        bottom=0.39,
        top=0.94
    )

    renderer = FieldRenderer(
        ax,
        domain
    )

    dynamic_renderer = DynamicRenderer(
        ax,
        domain
    )

    animation = AnimationEngine(
        fig,
        target_fps=60,
        text_fps=10
    )

    probe_text = fig.text(
        0.70,
        0.92,
        "Right-click to place probe",
        fontsize=10,
        va="top",
        linespacing=1.4
    )

    particle_text = fig.text(
        0.73,
        0.65,
        particle_status,
        fontsize=9,
        va="top",
        linespacing=1.1
    )

    def create_textbox(
        bounds,
        label,
        initial
    ):

        textbox_ax = fig.add_axes(
            bounds
        )

        textbox = TextBox(
            textbox_ax,
            label,
            initial=initial,
            color="#21262d",
            hovercolor="#30363d"
        )

        textbox.label.set_color("white")
        textbox.text_disp.set_color("white")

        return textbox

    fig.text(
        0.70,
        0.30,
        "INITIAL CONDITIONS",
        fontsize=10,
        fontweight="bold"
    )

    x_input = create_textbox(
        [0.73, 0.25, 0.09, 0.032],
        "x (m) ",
        "0"
    )

    y_input = create_textbox(
        [0.88, 0.25, 0.09, 0.032],
        "y (m) ",
        "2"
    )

    vx_input = create_textbox(
        [0.73, 0.20, 0.09, 0.032],
        "vx ",
        "0"
    )

    vy_input = create_textbox(
        [0.88, 0.20, 0.09, 0.032],
        "vy ",
        "1"
    )

    q_input = create_textbox(
        [0.73, 0.15, 0.09, 0.032],
        "q (nC) ",
        "1"
    )

    mass_input = create_textbox(
        [0.88, 0.15, 0.09, 0.032],
        "m (ug) ",
        "1"
    )

    def read_particle_parameters():

        values = np.array([
            float(x_input.text),
            float(y_input.text),
            float(vx_input.text),
            float(vy_input.text),
            float(q_input.text),
            float(mass_input.text)
        ])

        if not np.all(np.isfinite(values)):

            raise ValueError(
                "All parameters must be finite"
            )

        if values[5] <= 0:

            raise ValueError(
                "Mass must be positive"
            )

        return (
            values[:2],
            values[2:4],
            values[4] * 1e-9,
            values[5] * 1e-9
        )

    def refresh_particle(force=False):

        if particle_marker is None:

            return False

        if particle is None:

            particle_marker.set_data([], [])
            trajectory_line.set_data([], [])

            particle_text.set_text(
                particle_status
            )

            return True

        position = particle.get_position()
        velocity = particle.get_velocity()

        history = particle.trajectory

        stride = max(
            1,
            (len(history) + 3999) // 4000
        )

        trajectory = np.asarray(
            history[::stride]
        )

        if (len(history) - 1) % stride != 0:

            trajectory = np.vstack([
                trajectory,
                position
            ])

        particle_marker.set_data(
            [position[0]],
            [position[1]]
        )

        trajectory_line.set_data(
            trajectory[:, 0],
            trajectory[:, 1]
        )

        if not animation.should_update_text(force):

            return False

        try:

            kinetic, potential, total = (
                particle.get_energy(
                    charges,
                    positions
                )
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
            f"FPS = {animation.get_fps():.1f}\n"
            f"Render = {animation.get_render_time():.2f} ms\n"
            f"Physics = {integration_time:.2f} ms\n"
            f"{particle_status}"
        )

        return True

    def draw_probe():

        if probe_point is None:

            probe_text.set_text(
                "Right-click to place probe"
            )

            return

        ax.scatter(
            probe_point[0],
            probe_point[1],
            marker="x",
            color="yellow",
            s=80,
            zorder=12
        )

        try:

            field, magnitude, potential = (
                calculate_probe_values(
                    charges,
                    positions,
                    probe_point
                )
            )

            probe_text.set_text(
                "FIELD PROBE\n\n"
                f"x = {probe_point[0]:.3f} m\n"
                f"y = {probe_point[1]:.3f} m\n\n"
                f"Ex = {field[0]:.4g} N/C\n"
                f"Ey = {field[1]:.4g} N/C\n"
                f"|E| = {magnitude:.4g} N/C\n"
                f"V = {potential:.4g} V"
            )

            if magnitude > 0 and np.isfinite(magnitude):

                direction = field / magnitude

                length = domain.limit * 0.13

                ax.arrow(
                    probe_point[0],
                    probe_point[1],
                    direction[0] * length,
                    direction[1] * length,
                    color="yellow",
                    width=domain.limit * 0.004,
                    head_width=domain.limit * 0.035,
                    length_includes_head=True,
                    zorder=12
                )

        except ValueError:

            probe_text.set_text(
                "FIELD PROBE\n\n"
                "Undefined at point charge"
            )

    def create_particle_artists():

        nonlocal particle_marker
        nonlocal trajectory_line

        trajectory_line, = ax.plot(
            [],
            [],
            color="#a3e635",
            linewidth=1.5,
            zorder=8
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

    def update_field():

        nonlocal preview_active
        nonlocal preview_dirty
        nonlocal field_dirty

        preview_active = False
        preview_dirty = False
        field_dirty = False

        animation.set_artists()

        renderer.render(
            charges,
            positions,
            selected_charge,
            view_modes[view_index],
            fast=False
        )

        draw_probe()

        create_particle_artists()

        animation.set_artists(
            trajectory_line,
            particle_marker,
            particle_text
        )

        refresh_particle(force=True)

        fig.canvas.draw()

    def begin_preview():

        nonlocal preview_active
        nonlocal preview_dirty
        nonlocal field_dirty
        nonlocal last_preview_update

        animation.set_artists()

        dynamic_renderer.initialize(
            charges,
            positions,
            selected_charge
        )

        create_particle_artists()

        animation.set_artists(
            *dynamic_renderer.get_artists(),
            trajectory_line,
            particle_marker,
            particle_text
        )

        preview_active = True
        preview_dirty = False
        field_dirty = False

        if probe_point is not None:

            probe_text.set_text(
                "FIELD PROBE\n\n"
                "Updating field..."
            )

        refresh_particle(force=True)

        last_preview_update = time.perf_counter()

        fig.canvas.draw()

    def update_preview(force=False):

        nonlocal last_preview_update
        nonlocal preview_dirty
        nonlocal field_dirty

        if not preview_active:

            begin_preview()

            return

        now = time.perf_counter()

        changed = False

        if preview_dirty or force:

            dynamic_renderer.update_charges(
                charges,
                positions,
                selected_charge
            )

            preview_dirty = False
            changed = True

        if field_dirty or force:

            if force or (
                now - last_preview_update
                >= preview_interval
            ):

                dynamic_renderer.update_field(
                    charges,
                    positions
                )

                last_preview_update = now

                field_dirty = False
                changed = True

        if changed:

            animation.render(
                update_text=False
            )

    def pause_particle(status="Paused"):

        nonlocal particle_status

        if particle is not None:

            particle.pause()

        particle_status = status

        start_button.label.set_text(
            "Start"
        )

    def create_particle(position=None):

        nonlocal particle
        nonlocal placing_particle
        nonlocal integration_time

        try:

            initial_position, velocity, q, mass = (
                read_particle_parameters()
            )

            if position is not None:

                initial_position = np.asarray(
                    position,
                    dtype=float
                )

            active_positions = positions[
                charges != 0
            ]

            distances = np.linalg.norm(
                active_positions - initial_position,
                axis=1
            )

            if np.any(distances < 0.08):

                raise ValueError(
                    "Position too close to a charge"
                )

            particle = ParticleSimulator(
                position=initial_position,
                velocity=velocity,
                particle_charge=q,
                particle_mass=mass,
                dt=0.001,
                method=methods[method_index]
            )

            placing_particle = False

            x_input.set_val(
                f"{initial_position[0]:.6g}"
            )

            y_input.set_val(
                f"{initial_position[1]:.6g}"
            )

            pause_particle("Ready")

            integration_time = 0.0

            animation.reset_fps()

            domain.expand_if_needed(
                initial_position
            )

            update_field()

            return True

        except ValueError as error:

            pause_particle(str(error))

            refresh_particle(force=True)

            fig.canvas.draw_idle()

            return False

    def apply_parameters(event):

        create_particle()

    def place_particle(event):

        nonlocal placing_particle

        pause_particle(
            "Click the field to place particle"
        )

        placing_particle = True

        refresh_particle(force=True)

        fig.canvas.draw_idle()

    def toggle_particle(event):

        nonlocal particle_status
        nonlocal placing_particle

        if particle is None:

            if not create_particle():

                return

        placing_particle = False

        if particle.running:

            pause_particle()

        else:

            particle.start()

            particle_status = "Running"

            start_button.label.set_text(
                "Pause"
            )

            animation.reset_fps()

        refresh_particle(force=True)

        fig.canvas.draw_idle()

    def reset_particle(event):

        nonlocal placing_particle
        nonlocal integration_time

        placing_particle = False

        if particle is not None:

            particle.reset()

            pause_particle("Ready")

        else:

            pause_particle(
                "Place a particle to begin"
            )

        integration_time = 0.0

        refresh_particle(force=True)

        fig.canvas.draw_idle()

    def change_integrator(event):

        nonlocal method_index

        method_index = (
            method_index + 1
        ) % len(methods)

        method = methods[method_index]

        if particle is not None:

            pause_particle(
                f"Integrator changed to {method}"
            )

            particle.method = method

        integrator_button.label.set_text(
            f"Integrator: {method}"
        )

        refresh_particle(force=True)

        fig.canvas.draw_idle()

    def animate_particle():

        nonlocal integration_time

        if dragging or slider_changing:

            if preview_dirty or field_dirty:

                update_preview()

            return

        if particle is None or not particle.running:

            return

        physics_start = time.perf_counter()

        active_positions = positions[
            charges != 0
        ]

        for _ in range(10):

            position = particle.get_position()

            distances = np.linalg.norm(
                active_positions - position,
                axis=1
            )

            if np.any(distances < 0.08):

                pause_particle(
                    "Near point charge: stopped"
                )

                break

            previous_state = particle.state.copy()

            previous_time = particle.time

            previous_length = len(
                particle.trajectory
            )

            try:

                particle.step(
                    charges,
                    positions
                )

                if not np.all(
                    np.isfinite(particle.state)
                ):

                    raise FloatingPointError()

            except (
                ValueError,
                FloatingPointError
            ):

                particle.state = previous_state

                particle.time = previous_time

                del particle.trajectory[
                    previous_length:
                ]

                pause_particle(
                    "Numerical singularity: stopped"
                )

                break

        elapsed = (
            time.perf_counter() - physics_start
        ) * 1000

        if integration_time == 0:

            integration_time = elapsed

        else:

            integration_time = (
                0.9 * integration_time
                + 0.1 * elapsed
            )

        if domain.expand_if_needed(
            particle.get_position()
        ):

            update_field()

            return

        text_updated = refresh_particle(
            force=not particle.running
        )

        animation.render(
            update_text=text_updated
        )

        if not particle.running:

            fig.canvas.draw_idle()

    def sync_slider():

        nonlocal syncing_slider

        syncing_slider = True

        if selected_charge is not None:

            charge_slider.set_active(True)

            charge_slider.set_val(
                charges[selected_charge] * 1e9
            )

        else:

            charge_slider.set_active(False)

        syncing_slider = False

    def update_charge(value):

        nonlocal slider_changing
        nonlocal preview_dirty
        nonlocal field_dirty

        if syncing_slider:

            return

        if selected_charge is None:

            return

        if not slider_changing:

            pause_particle(
                "Source charge modified"
            )

        charges[selected_charge] = value * 1e-9

        slider_changing = True

        preview_dirty = True
        field_dirty = True

    def on_press(event):

        nonlocal selected_charge
        nonlocal dragging
        nonlocal probe_point

        if event.inaxes is not ax:

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

            create_particle(position)

            return

        if len(positions) == 0:

            return

        distances = np.linalg.norm(
            positions - position,
            axis=1
        )

        nearest = np.argmin(distances)

        if distances[nearest] < domain.limit * 0.06:

            selected_charge = nearest

            pause_particle(
                "Moving source charge"
            )

            dragging = True

            sync_slider()

            begin_preview()

    def on_motion(event):

        nonlocal preview_dirty
        nonlocal field_dirty

        if not dragging:

            return

        if event.inaxes is not ax:

            return

        if event.xdata is None or event.ydata is None:

            return

        positions[selected_charge] = [
            event.xdata,
            event.ydata
        ]

        preview_dirty = True
        field_dirty = True

    def on_release(event):

        nonlocal dragging
        nonlocal slider_changing
        nonlocal preview_dirty
        nonlocal field_dirty

        if dragging or slider_changing:

            if dragging and event.inaxes is ax:

                if (
                    event.xdata is not None
                    and event.ydata is not None
                ):

                    positions[selected_charge] = [
                        event.xdata,
                        event.ydata
                    ]

            dragging = False
            slider_changing = False

            preview_dirty = False
            field_dirty = False

            update_field()

    def add_charge(value):

        nonlocal charges
        nonlocal positions
        nonlocal selected_charge

        pause_particle(
            "Source charge added"
        )

        position = np.array([
            0.0,
            0.0
        ])

        if len(positions):

            for _ in range(100):

                distances = np.linalg.norm(
                    positions - position,
                    axis=1
                )

                if np.all(distances > 0.5):

                    break

                position = np.random.uniform(
                    -domain.limit * 0.6,
                    domain.limit * 0.6,
                    size=2
                )

        charges = np.append(
            charges,
            value
        )

        positions = np.vstack([
            positions,
            position
        ])

        selected_charge = len(charges) - 1

        sync_slider()

        update_field()

    def remove_charge(event):

        nonlocal charges
        nonlocal positions
        nonlocal selected_charge

        if selected_charge is None:

            return

        pause_particle(
            "Source charge removed"
        )

        charges = np.delete(
            charges,
            selected_charge
        )

        positions = np.delete(
            positions,
            selected_charge,
            axis=0
        )

        if len(charges):

            selected_charge = min(
                selected_charge,
                len(charges) - 1
            )

        else:

            selected_charge = None

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

    def clear_probe(event):

        nonlocal probe_point

        probe_point = None

        update_field()

    def reset(event):

        nonlocal charges
        nonlocal positions
        nonlocal selected_charge
        nonlocal particle
        nonlocal probe_point
        nonlocal placing_particle
        nonlocal dragging
        nonlocal slider_changing
        nonlocal preview_dirty
        nonlocal field_dirty
        nonlocal integration_time

        pause_particle()

        charges = np.array([
            5e-9,
            -5e-9
        ])

        positions = np.array([
            [-2.0, 0.0],
            [2.0, 0.0]
        ])

        selected_charge = 0

        particle = None
        probe_point = None

        placing_particle = False
        dragging = False
        slider_changing = False
        preview_dirty = False
        field_dirty = False

        integration_time = 0.0

        domain.reset()

        pause_particle(
            "Place a particle to begin"
        )

        sync_slider()

        animation.reset_fps()

        update_field()

    def create_button(
        bounds,
        label,
        callback,
        color="#30363d"
    ):

        button_ax = fig.add_axes(
            bounds
        )

        button = Button(
            button_ax,
            label,
            color=color,
            hovercolor="#555555"
        )

        button.label.set_color("white")

        button.on_clicked(callback)

        return button

    slider_ax = fig.add_axes([
        0.22,
        0.30,
        0.41,
        0.025
    ])

    charge_slider = Slider(
        slider_ax,
        "Selected charge (nC)",
        -10,
        10,
        valinit=5,
        valstep=0.1
    )

    positive_button = create_button(
        [0.07, 0.23, 0.135, 0.04],
        "+ Positive",
        lambda event: add_charge(5e-9),
        "#54252b"
    )

    negative_button = create_button(
        [0.215, 0.23, 0.135, 0.04],
        "+ Negative",
        lambda event: add_charge(-5e-9),
        "#163d52"
    )

    remove_button = create_button(
        [0.36, 0.23, 0.135, 0.04],
        "Remove",
        remove_charge
    )

    reset_button = create_button(
        [0.505, 0.23, 0.135, 0.04],
        "Reset",
        reset
    )

    view_button = create_button(
        [0.07, 0.16, 0.35, 0.04],
        "View: Electric Field",
        change_view,
        "#35305c"
    )

    probe_button = create_button(
        [0.43, 0.16, 0.21, 0.04],
        "Clear Probe",
        clear_probe
    )

    place_button = create_button(
        [0.07, 0.09, 0.135, 0.04],
        "Place Particle",
        place_particle
    )

    start_button = create_button(
        [0.215, 0.09, 0.135, 0.04],
        "Start",
        toggle_particle,
        "#24543b"
    )

    reset_particle_button = create_button(
        [0.36, 0.09, 0.135, 0.04],
        "Reset Particle",
        reset_particle
    )

    integrator_button = create_button(
        [0.505, 0.09, 0.135, 0.04],
        "Integrator: RK4",
        change_integrator,
        "#35305c"
    )

    apply_button = create_button(
        [0.73, 0.09, 0.24, 0.04],
        "Apply Initial Conditions",
        apply_parameters,
        "#24543b"
    )

    charge_slider.on_changed(
        update_charge
    )

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
        interval=animation.get_timer_interval()
    )

    timer.add_callback(
        animate_particle
    )

    fig.canvas.mpl_connect(
        "close_event",
        lambda event: timer.stop()
    )

    update_field()

    timer.start()

    plt.show()


if __name__ == "__main__":

    run_interactive_simulator()