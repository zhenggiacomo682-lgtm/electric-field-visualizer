import time
import numpy as np
import pyqtgraph as pg

from PySide6 import QtCore, QtWidgets
from src.particle_simulator import ParticleSimulator
from src.field_probe import calculate_probe_values

pg.setConfigOptions(
    antialias=False,
    background="#0d1117",
    foreground="#dce3eb",
    imageAxisOrder="row-major"
)


class SimulationView(pg.ViewBox):

    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.dragged_charge = None

    def mouseDragEvent(self, event, axis=None):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(event, axis)
            return

        if event.isStart():
            point = self.mapSceneToView(event.buttonDownScenePos())
            self.dragged_charge = self.simulator.find_charge(
                point.x(), point.y()
            )

            if self.dragged_charge is not None:
                self.simulator.begin_drag(self.dragged_charge)

        if self.dragged_charge is None:
            super().mouseDragEvent(event, axis)
            return

        event.accept()
        point = self.mapSceneToView(event.scenePos())

        self.simulator.move_charge(
            self.dragged_charge, point.x(), point.y()
        )

        if event.isFinish():
            self.simulator.end_drag()
            self.dragged_charge = None


class ElectricFieldSimulator(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Electric Field Simulator")
        self.resize(1500, 900)

        self.charges = np.array([5e-9, -5e-9])
        self.positions = np.array([[-2.0, 0.0], [2.0, 0.0]])

        self.selected_charge = 0
        self.domain = 5.0
        self.particle = None
        self.probe_point = None

        self.dragging = False
        self.placing_particle = False
        self.field_dirty = False

        self.last_field_update = 0.0
        self.last_text_update = 0.0
        self.last_fps_update = time.perf_counter()

        self.frame_count = 0
        self.fps = 0.0
        self.physics_time = 0.0
        self.render_time = 0.0

        self.field_cache_key = None
        self.potential_cache_key = None

        self.status = "Ready"

        self.build_interface()

        self.range_timer = QtCore.QTimer(self)
        self.range_timer.setSingleShot(True)
        self.range_timer.timeout.connect(self.refresh_visible_field)

        self.view.sigRangeChanged.connect(self.schedule_range_update)

        self.update_sources()
        self.sync_source_inputs()
        self.update_field()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)

    def build_interface(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        self.setStyleSheet("""
            QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-size: 13px;
            }

            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }

            QPushButton {
                background-color: #21262d;
                border: 1px solid #484f58;
                border-radius: 5px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #30363d;
            }

            QPushButton:pressed {
                background-color: #484f58;
            }

            QDoubleSpinBox, QComboBox {
                background-color: #161b22;
                border: 1px solid #484f58;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }

            QLabel {
                color: #e6edf3;
            }
        """)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        layout.addWidget(left_panel, 4)

        self.view = SimulationView(self)
        self.plot = pg.PlotWidget(viewBox=self.view)

        self.plot.setTitle(
            "Electric Field Simulator",
            color="#ffffff",
            size="16pt"
        )

        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")

        self.plot.showGrid(x=True, y=True, alpha=0.10)
        self.plot.setAspectLocked(True)
        self.plot.disableAutoRange()

        self.plot.setRange(
            xRange=(-5, 5),
            yRange=(-5, 5),
            padding=0
        )

        left_layout.addWidget(self.plot, 1)

        self.potential_image = pg.ImageItem()
        self.potential_image.setZValue(-10)
        self.plot.addItem(self.potential_image)

        self.potential_colormap = pg.ColorMap(
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([
                [30, 95, 220],
                [20, 45, 100],
                [13, 17, 23],
                [110, 30, 30],
                [245, 75, 65]
            ], dtype=np.ubyte)
        )

        self.potential_image.setLookupTable(
            self.potential_colormap.getLookupTable(nPts=256)
        )

        self.potential_image.hide()

        self.contours = []

        for _ in range(13):
            contour = pg.IsocurveItem(
                level=0,
                pen=pg.mkPen("#94a3b8", width=1),
                axisOrder="row-major"
            )

            contour.setParentItem(self.potential_image)
            contour.hide()
            self.contours.append(contour)

        self.field_colors = [
            "#4145ab",
            "#3865df",
            "#2697f0",
            "#22c7df",
            "#35dbad",
            "#68df65",
            "#b5e346",
            "#f2df38",
            "#ffb52e",
            "#ff8130",
            "#f54b35",
            "#df2935"
        ]

        self.field_curves = []

        for color in self.field_colors:
            curve = pg.PlotDataItem(
                pen=pg.mkPen(color, width=1.35),
                connect="finite",
                antialias=True
            )

            curve.setZValue(1)
            self.plot.addItem(curve)
            self.field_curves.append(curve)

        self.field_arrows = pg.PlotDataItem(
            pen=pg.mkPen("#b8e4ed", width=1.2),
            connect="finite",
            antialias=True
        )

        self.field_arrows.setZValue(2)
        self.plot.addItem(self.field_arrows)

        self.vector_preview = pg.PlotDataItem(
            pen=pg.mkPen("#38bdf8", width=1),
            connect="finite"
        )

        self.vector_preview.setZValue(2)
        self.plot.addItem(self.vector_preview)
        self.vector_preview.hide()

        self.source_markers = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen("white", width=1.5)
        )

        self.source_markers.setZValue(10)
        self.plot.addItem(self.source_markers)

        self.selection_marker = pg.ScatterPlotItem(
            size=25,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen("#facc15", width=2)
        )

        self.selection_marker.setZValue(11)
        self.plot.addItem(self.selection_marker)

        self.source_labels = []

        self.trajectory_item = pg.PlotDataItem(
            pen=pg.mkPen("#a3e635", width=1.8),
            antialias=True
        )

        self.trajectory_item.setZValue(5)
        self.plot.addItem(self.trajectory_item)

        self.particle_item = pg.ScatterPlotItem(
            size=11,
            brush=pg.mkBrush("#a3e635"),
            pen=pg.mkPen("white", width=1)
        )

        self.particle_item.setZValue(12)
        self.plot.addItem(self.particle_item)

        self.probe_item = pg.ScatterPlotItem(
            size=13,
            symbol="x",
            brush=pg.mkBrush("#facc15"),
            pen=pg.mkPen("#facc15")
        )

        self.probe_item.setZValue(13)
        self.plot.addItem(self.probe_item)

        self.probe_vector = pg.PlotDataItem(
            pen=pg.mkPen("#facc15", width=2),
            connect="finite",
            antialias=True
        )

        self.probe_vector.setZValue(13)
        self.plot.addItem(self.probe_vector)

        self.plot.scene().sigMouseClicked.connect(self.on_plot_click)

        controls = QtWidgets.QHBoxLayout()

        self.add_button(
            controls, "+ Positive",
            lambda: self.add_charge(5e-9), "#54252b"
        )

        self.add_button(
            controls, "+ Negative",
            lambda: self.add_charge(-5e-9), "#163d52"
        )

        self.add_button(controls, "Remove Charge", self.remove_charge)
        self.add_button(controls, "Reset All", self.reset_all)

        left_layout.addLayout(controls)

        view_controls = QtWidgets.QHBoxLayout()

        self.view_mode = QtWidgets.QComboBox()

        self.view_mode.addItems([
            "Electric Field",
            "Equipotential",
            "Combined"
        ])

        self.view_mode.currentTextChanged.connect(self.change_view)

        view_controls.addWidget(self.view_mode)

        self.add_button(view_controls, "Fit Domain", self.fit_domain)
        self.add_button(view_controls, "Clear Probe", self.clear_probe)

        left_layout.addLayout(view_controls)

        self.color_label = QtWidgets.QLabel(
            "Field magnitude: blue (weak) → red (strong)"
        )

        self.color_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )

        left_layout.addWidget(self.color_label)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(420)
        scroll.setWidget(right_panel)

        layout.addWidget(scroll, 1)

        source_group = QtWidgets.QGroupBox("SOURCE CHARGE")
        source_form = QtWidgets.QFormLayout(source_group)

        self.source_x = self.create_spinbox(-2, -10000, 10000)
        self.source_y = self.create_spinbox(0, -10000, 10000)
        self.source_q = self.create_spinbox(5, -1000, 1000)

        source_form.addRow("x (m)", self.source_x)
        source_form.addRow("y (m)", self.source_y)
        source_form.addRow("q (nC)", self.source_q)

        self.add_button(
            source_form,
            "Apply Source Values",
            self.apply_source
        )

        right_layout.addWidget(source_group)

        particle_group = QtWidgets.QGroupBox("TEST PARTICLE")
        particle_form = QtWidgets.QFormLayout(particle_group)

        self.particle_x = self.create_spinbox(0, -10000, 10000)
        self.particle_y = self.create_spinbox(2, -10000, 10000)

        self.particle_vx = self.create_spinbox(
            0, -100000, 100000
        )

        self.particle_vy = self.create_spinbox(
            1, -100000, 100000
        )

        self.particle_q = self.create_spinbox(1, -1000, 1000)

        self.particle_mass = self.create_spinbox(
            1, 0.000001, 1000000, decimals=6
        )

        particle_form.addRow("x (m)", self.particle_x)
        particle_form.addRow("y (m)", self.particle_y)
        particle_form.addRow("vx (m/s)", self.particle_vx)
        particle_form.addRow("vy (m/s)", self.particle_vy)
        particle_form.addRow("q (nC)", self.particle_q)
        particle_form.addRow("m (µg)", self.particle_mass)

        self.integrator = QtWidgets.QComboBox()

        self.integrator.addItems(["Euler", "RK4", "Verlet"])
        self.integrator.setCurrentText("RK4")

        self.integrator.currentTextChanged.connect(
            self.change_integrator
        )

        particle_form.addRow("Integrator", self.integrator)

        self.add_button(
            particle_form,
            "Apply Initial Conditions",
            self.create_particle
        )

        self.add_button(
            particle_form,
            "Place With Mouse",
            self.place_particle
        )

        self.start_button = self.add_button(
            particle_form,
            "Start Simulation",
            self.toggle_particle,
            "#24543b"
        )

        self.add_button(
            particle_form,
            "Reset Particle",
            self.reset_particle
        )

        right_layout.addWidget(particle_group)

        probe_group = QtWidgets.QGroupBox("FIELD PROBE")
        probe_layout = QtWidgets.QVBoxLayout(probe_group)

        self.probe_label = QtWidgets.QLabel(
            "Right-click to place probe"
        )

        probe_layout.addWidget(self.probe_label)
        right_layout.addWidget(probe_group)

        data_group = QtWidgets.QGroupBox("SIMULATION DATA")
        data_layout = QtWidgets.QVBoxLayout(data_group)

        self.data_label = QtWidgets.QLabel("No particle")
        self.data_label.setStyleSheet("font-family: Consolas;")

        data_layout.addWidget(self.data_label)
        right_layout.addWidget(data_group)

        self.performance_label = QtWidgets.QLabel("Update FPS: 0")
        right_layout.addWidget(self.performance_label)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)

        right_layout.addWidget(self.status_label)
        right_layout.addStretch()

    def create_spinbox(
        self, value, minimum, maximum, decimals=4
    ):
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        spinbox.setSingleStep(0.1)
        spinbox.setKeyboardTracking(False)
        return spinbox

    def add_button(self, layout, text, callback, color=None):
        button = QtWidgets.QPushButton(text)

        button.clicked.connect(
            lambda checked=False: callback()
        )

        if color is not None:
            button.setStyleSheet(
                f"background-color: {color};"
            )

        if isinstance(layout, QtWidgets.QFormLayout):
            layout.addRow(button)
        else:
            layout.addWidget(button)

        return button

    def set_status(self, status):
        self.status = status
        self.status_label.setText(status)

    def visible_bounds(self):
        ranges = self.view.viewRange()

        return (
            ranges[0][0],
            ranges[0][1],
            ranges[1][0],
            ranges[1][1]
        )

    def cache_key(self):
        bounds = self.visible_bounds()

        return (
            self.charges.tobytes(),
            self.positions.tobytes(),
            tuple(round(value, 7) for value in bounds)
        )

    def invalidate_cache(self):
        self.field_cache_key = None
        self.potential_cache_key = None

    def schedule_range_update(self, *args):
        self.range_timer.start(100)

    def refresh_visible_field(self):
        self.update_field(fast=self.dragging)

    def calculate_field(self, points):
        points = np.asarray(points, dtype=float)
        field = np.zeros_like(points)

        for charge, position in zip(
            self.charges, self.positions
        ):
            if charge == 0:
                continue

            displacement = points - position

            distance_squared = np.sum(
                displacement ** 2, axis=-1
            )

            safe_distance = np.maximum(
                distance_squared, 1e-12
            )

            field += (
                8.9875517923e9
                * charge
                * displacement
                / safe_distance[..., None] ** 1.5
            )

        return field

    def calculate_potential(self, points):
        points = np.asarray(points, dtype=float)

        potential = np.zeros(points.shape[:-1])

        for charge, position in zip(
            self.charges, self.positions
        ):
            if charge == 0:
                continue

            displacement = points - position

            distance = np.linalg.norm(
                displacement, axis=-1
            )

            potential += (
                8.9875517923e9
                * charge
                / np.maximum(distance, 1e-6)
            )

        return potential

    def normalized_field(self, points):
        field = self.calculate_field(points)

        magnitude = np.linalg.norm(
            field, axis=-1, keepdims=True
        )

        return np.divide(
            field,
            magnitude,
            out=np.zeros_like(field),
            where=magnitude > 1e-12
        )

    def create_seeds(self):
        xmin, xmax, ymin, ymax = self.visible_bounds()

        span = min(xmax - xmin, ymax - ymin)

        radius = max(span * 0.018, 0.015)

        seeds = []
        directions = []

        active = self.charges != 0

        positions = self.positions[active]
        charges = self.charges[active]

        if not len(charges):
            return np.empty((0, 2)), np.empty(0)

        maximum_charge = np.max(np.abs(charges))

        for charge, position in zip(charges, positions):
            if charge < 0 and np.any(charges > 0):
                continue

            count = int(
                np.clip(
                    24 * np.sqrt(abs(charge) / maximum_charge),
                    8,
                    32
                )
            )

            angles = np.linspace(
                0, 2 * np.pi, count, endpoint=False
            )

            offsets = np.column_stack([
                np.cos(angles),
                np.sin(angles)
            ])

            source_seeds = position + radius * offsets

            seeds.extend(source_seeds)

            directions.extend(
                [1 if charge > 0 else -1] * count
            )

        margin = span * 0.015

        x_values = np.linspace(
            xmin + margin, xmax - margin, 13
        )

        y_values = np.linspace(
            ymin + margin, ymax - margin, 13
        )

        boundary = []

        for x in x_values:
            boundary.extend([
                [x, ymin + margin],
                [x, ymax - margin]
            ])

        for y in y_values[1:-1]:
            boundary.extend([
                [xmin + margin, y],
                [xmax - margin, y]
            ])

        boundary = np.asarray(boundary)

        field = self.normalized_field(boundary)

        center = np.array([
            (xmin + xmax) / 2,
            (ymin + ymax) / 2
        ])

        inward = center - boundary

        signs = np.where(
            np.sum(field * inward, axis=1) >= 0,
            1,
            -1
        )

        seeds.extend(boundary)
        directions.extend(signs)

        seeds = np.asarray(seeds)
        directions = np.asarray(directions)

        inside = (
            (seeds[:, 0] >= xmin)
            & (seeds[:, 0] <= xmax)
            & (seeds[:, 1] >= ymin)
            & (seeds[:, 1] <= ymax)
        )

        return seeds[inside], directions[inside]

    def trace_lines(self, seeds, directions):
        xmin, xmax, ymin, ymax = self.visible_bounds()

        span = min(xmax - xmin, ymax - ymin)

        step = span / 170

        points = seeds.copy()

        active = np.ones(len(seeds), dtype=bool)

        paths = [points.copy()]

        active_positions = self.positions[
            self.charges != 0
        ]

        directions = directions[:, None]

        for _ in range(340):
            if not np.any(active):
                break

            indices = np.flatnonzero(active)
            current = points[indices]
            signs = directions[indices]

            first = self.normalized_field(current)

            middle = current + signs * step * first / 2

            second = self.normalized_field(middle)

            candidate = current + signs * step * second

            inside = (
                (candidate[:, 0] >= xmin)
                & (candidate[:, 0] <= xmax)
                & (candidate[:, 1] >= ymin)
                & (candidate[:, 1] <= ymax)
            )

            moving = (
                np.linalg.norm(second, axis=1) > 1e-10
            )

            distances = np.linalg.norm(
                candidate[:, None, :]
                - active_positions[None, :, :],
                axis=2
            )

            safe = np.all(
                distances > step * 1.3,
                axis=1
            )

            valid = inside & moving & safe

            active[indices[~valid]] = False

            points[indices[valid]] = candidate[valid]

            frame = np.full_like(points, np.nan)
            frame[active] = points[active]

            paths.append(frame)

        return np.asarray(paths)

    def create_arrow_segments(self, points, directions, length):
        points = np.asarray(points)
        directions = np.asarray(directions)

        perpendicular = np.column_stack([
            -directions[:, 1],
            directions[:, 0]
        ])

        left = (
            points
            - directions * length
            + perpendicular * length * 0.45
        )

        right = (
            points
            - directions * length
            - perpendicular * length * 0.45
        )

        segments = np.full(
            (len(points), 4, 2), np.nan
        )

        segments[:, 0] = left
        segments[:, 1] = points
        segments[:, 2] = right

        return segments.reshape(-1, 2)

    def update_streamlines(self):
        key = self.cache_key()

        if key == self.field_cache_key:
            return

        if not np.any(self.charges != 0):
            for curve in self.field_curves:
                curve.setData([], [])

            self.field_arrows.setData([], [])
            self.field_cache_key = key
            return

        seeds, directions = self.create_seeds()

        if not len(seeds):
            for curve in self.field_curves:
                curve.setData([], [])

            self.field_arrows.setData([], [])
            self.field_cache_key = key
            return

        paths = self.trace_lines(seeds, directions)

        xmin, xmax, ymin, ymax = self.visible_bounds()

        span = min(xmax - xmin, ymax - ymin)

        accepted_paths = []
        occupied = set()

        spacing = span / 65

        for index in range(paths.shape[1]):
            path = paths[:, index, :]

            path = path[
                np.all(np.isfinite(path), axis=1)
            ]

            if len(path) < 5:
                continue

            if directions[index] < 0:
                path = path[::-1]

            cells = np.floor(
                (path - np.array([xmin, ymin])) / spacing
            ).astype(int)

            sampled_cells = {
                tuple(cell) for cell in cells[::3]
            }

            overlap = len(
                sampled_cells.intersection(occupied)
            ) / max(len(sampled_cells), 1)

            if overlap > 0.75:
                continue

            occupied.update(sampled_cells)
            accepted_paths.append(path)

        if not accepted_paths:
            for curve in self.field_curves:
                curve.setData([], [])

            self.field_arrows.setData([], [])
            self.field_cache_key = key
            return

        magnitudes = []

        for path in accepted_paths:
            magnitude = np.linalg.norm(
                self.calculate_field(path),
                axis=1
            )

            magnitudes.append(magnitude)

        all_magnitudes = np.concatenate(magnitudes)

        reference = max(
            np.percentile(
                np.log1p(all_magnitudes), 95
            ),
            1e-12
        )

        color_groups = [
            [] for _ in self.field_colors
        ]

        arrow_points = []

        for path, magnitude in zip(
            accepted_paths, magnitudes
        ):
            color_values = np.clip(
                (
                    np.log1p(magnitude)
                    / reference
                    * len(self.field_colors)
                ).astype(int),
                0,
                len(self.field_colors) - 1
            )

            start = 0

            for index in range(1, len(path)):
                if color_values[index] != color_values[start]:
                    color_groups[
                        color_values[start]
                    ].append(path[start:index + 1])

                    start = index

            if start < len(path) - 1:
                color_groups[
                    color_values[start]
                ].append(path[start:])

            if len(path) >= 18:
                indices = np.arange(
                    12, len(path) - 5, 42
                )

                arrow_points.extend(path[indices])

        for curve, group in zip(
            self.field_curves, color_groups
        ):
            if not group:
                curve.setData([], [])
                continue

            pieces = []

            for path in group:
                pieces.append(path)
                pieces.append(
                    np.array([[np.nan, np.nan]])
                )

            coordinates = np.concatenate(pieces)

            curve.setData(
                coordinates[:, 0],
                coordinates[:, 1],
                connect="finite"
            )

        if arrow_points:
            arrow_points = np.asarray(arrow_points)

            directions = self.normalized_field(
                arrow_points
            )

            arrows = self.create_arrow_segments(
                arrow_points,
                directions,
                span * 0.012
            )

            self.field_arrows.setData(
                arrows[:, 0],
                arrows[:, 1],
                connect="finite"
            )
        else:
            self.field_arrows.setData([], [])

        self.field_cache_key = key

    def update_vector_preview(self):
        xmin, xmax, ymin, ymax = self.visible_bounds()

        span = min(xmax - xmin, ymax - ymin)

        spacing = span / 20

        nx = int(
            np.clip((xmax - xmin) / spacing, 10, 40)
        )

        ny = int(
            np.clip((ymax - ymin) / spacing, 10, 40)
        )

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        X, Y = np.meshgrid(x, y)

        points = np.column_stack([
            X.ravel(),
            Y.ravel()
        ])

        field = self.calculate_field(points)

        magnitude = np.linalg.norm(
            field, axis=1, keepdims=True
        )

        directions = np.divide(
            field,
            magnitude,
            out=np.zeros_like(field),
            where=magnitude > 1e-12
        )

        length = spacing * 0.65

        tips = points + directions * length

        perpendicular = np.column_stack([
            -directions[:, 1],
            directions[:, 0]
        ])

        left = (
            tips
            - directions * length * 0.35
            + perpendicular * length * 0.22
        )

        right = (
            tips
            - directions * length * 0.35
            - perpendicular * length * 0.22
        )

        segments = np.full(
            (len(points), 7, 2), np.nan
        )

        segments[:, 0] = points
        segments[:, 1] = tips
        segments[:, 3] = left
        segments[:, 4] = tips
        segments[:, 5] = right

        segments = segments.reshape(-1, 2)

        self.vector_preview.setData(
            segments[:, 0],
            segments[:, 1],
            connect="finite"
        )

    def update_potential(self, fast=False):
        key = self.cache_key()

        if not fast and key == self.potential_cache_key:
            return

        xmin, xmax, ymin, ymax = self.visible_bounds()

        resolution = 55 if fast else 180

        dx = (xmax - xmin) / resolution
        dy = (ymax - ymin) / resolution

        x = np.linspace(
            xmin + dx / 2,
            xmax - dx / 2,
            resolution
        )

        y = np.linspace(
            ymin + dy / 2,
            ymax - dy / 2,
            resolution
        )

        X, Y = np.meshgrid(x, y)

        points = np.stack([X, Y], axis=-1)

        potential = self.calculate_potential(points)

        limit = max(
            np.percentile(np.abs(potential), 98),
            1e-9
        )

        self.potential_image.setImage(
            potential,
            autoLevels=False,
            levels=(-limit, limit)
        )

        self.potential_image.setRect(
            QtCore.QRectF(
                xmin,
                ymin,
                xmax - xmin,
                ymax - ymin
            )
        )

        self.potential_image.setOpacity(0.85)

        if fast:
            for contour in self.contours:
                contour.hide()

            self.potential_cache_key = None
            return

        levels = limit * np.array([
            -0.95,
            -0.70,
            -0.50,
            -0.32,
            -0.18,
            -0.08,
            0.0,
            0.08,
            0.18,
            0.32,
            0.50,
            0.70,
            0.95
        ])

        for contour, level in zip(
            self.contours, levels
        ):
            if level < 0:
                color = "#82aaff"
            elif level > 0:
                color = "#ff9990"
            else:
                color = "#dce3eb"

            contour.setPen(
                pg.mkPen(
                    color,
                    width=1,
                    style=QtCore.Qt.PenStyle.DashLine
                )
            )

            contour.setData(potential)
            contour.setLevel(level)

            contour.setVisible(
                np.min(potential) < level < np.max(potential)
            )

        self.potential_cache_key = key

        self.color_label.setText(
            f"Potential scale: {-limit:.3g} V"
            f"  |  0 V  |  +{limit:.3g} V"
        )

    def update_field(self, fast=False):
        start = time.perf_counter()

        mode = self.view_mode.currentText()

        show_field = mode in [
            "Electric Field", "Combined"
        ]

        show_potential = mode in [
            "Equipotential", "Combined"
        ]

        self.potential_image.setVisible(show_potential)

        for curve in self.field_curves:
            curve.setVisible(show_field and not fast)

        self.field_arrows.setVisible(
            show_field and not fast
        )

        self.vector_preview.setVisible(
            show_field and fast
        )

        if show_field:
            if fast:
                self.update_vector_preview()
            else:
                self.update_streamlines()

        if show_potential:
            self.update_potential(fast)

        if mode == "Electric Field":
            self.color_label.setText(
                "Field magnitude: blue (weak) → red (strong)"
            )

        elif mode == "Combined":
            self.color_label.setText(
                "Colored field lines | Red: +V | Blue: −V"
            )

        self.field_dirty = False

        self.last_field_update = time.perf_counter()

        self.render_time = (
            time.perf_counter() - start
        ) * 1000

        self.update_probe()
        self.update_performance_label()

    def update_sources(self):
        colors = [
            "#ff4757" if charge > 0
            else "#38bdf8" if charge < 0
            else "#888888"
            for charge in self.charges
        ]

        self.source_markers.setData(
            x=self.positions[:, 0],
            y=self.positions[:, 1],
            brush=[pg.mkBrush(color) for color in colors]
        )

        if self.selected_charge is None:
            self.selection_marker.setData([], [])
        else:
            position = self.positions[self.selected_charge]

            self.selection_marker.setData(
                [position[0]], [position[1]]
            )

        while len(self.source_labels) > len(self.charges):
            label = self.source_labels.pop()
            self.plot.removeItem(label)

        while len(self.source_labels) < len(self.charges):
            label = pg.TextItem(
                anchor=(0.5, 1.5),
                fill=pg.mkBrush(13, 17, 23, 220)
            )

            label.setZValue(12)
            self.plot.addItem(label)
            self.source_labels.append(label)

        for label, charge, position, color in zip(
            self.source_labels,
            self.charges,
            self.positions,
            colors
        ):
            label.setText(
                f"{charge * 1e9:+.2f} nC",
                color=color
            )

            label.setPos(position[0], position[1])

    def find_charge(self, x, y):
        if not len(self.positions):
            return None

        point = np.array([x, y])

        distances = np.linalg.norm(
            self.positions - point, axis=1
        )

        nearest = np.argmin(distances)

        pixel_size = max(self.view.viewPixelSize())

        if distances[nearest] <= pixel_size * 15:
            return int(nearest)

        return None

    def sync_source_inputs(self):
        enabled = self.selected_charge is not None

        for widget in [
            self.source_x,
            self.source_y,
            self.source_q
        ]:
            widget.setEnabled(enabled)

        if not enabled:
            return

        index = self.selected_charge

        self.source_x.setValue(self.positions[index, 0])
        self.source_y.setValue(self.positions[index, 1])
        self.source_q.setValue(self.charges[index] * 1e9)

    def apply_source(self):
        if self.selected_charge is None:
            return

        self.pause_particle("Source charge modified")

        index = self.selected_charge

        self.positions[index] = [
            self.source_x.value(),
            self.source_y.value()
        ]

        self.charges[index] = (
            self.source_q.value() * 1e-9
        )

        self.invalidate_cache()
        self.expand_domain(self.positions[index])
        self.update_sources()
        self.update_field()
        self.update_particle_data()

    def begin_drag(self, index):
        self.selected_charge = index
        self.dragging = True

        self.pause_particle("Moving source charge")

        self.sync_source_inputs()
        self.update_sources()
        self.update_field(fast=True)

    def move_charge(self, index, x, y):
        self.positions[index] = [x, y]

        self.field_dirty = True
        self.invalidate_cache()

        self.update_sources()

    def end_drag(self):
        self.dragging = False

        self.sync_source_inputs()

        if self.selected_charge is not None:
            self.expand_domain(
                self.positions[self.selected_charge]
            )

        self.update_field()
        self.update_particle_data()
        self.set_status("Source charge moved")

    def add_charge(self, charge):
        self.pause_particle("Source charge added")

        position = np.array([0.0, 0.0])

        for _ in range(100):
            if not len(self.positions):
                break

            distances = np.linalg.norm(
                self.positions - position, axis=1
            )

            if np.all(distances > 0.5):
                break

            position = np.random.uniform(
                -self.domain * 0.5,
                self.domain * 0.5,
                size=2
            )

        self.charges = np.append(
            self.charges, charge
        )

        self.positions = np.vstack([
            self.positions, position
        ])

        self.selected_charge = len(self.charges) - 1

        self.invalidate_cache()
        self.sync_source_inputs()
        self.update_sources()
        self.update_field()
        self.update_particle_data()

    def remove_charge(self):
        if self.selected_charge is None:
            return

        self.pause_particle("Source charge removed")

        self.charges = np.delete(
            self.charges, self.selected_charge
        )

        self.positions = np.delete(
            self.positions,
            self.selected_charge,
            axis=0
        )

        if len(self.charges):
            self.selected_charge = min(
                self.selected_charge,
                len(self.charges) - 1
            )
        else:
            self.selected_charge = None

        self.invalidate_cache()
        self.sync_source_inputs()
        self.update_sources()
        self.update_field()
        self.update_particle_data()

    def change_view(self, mode):
        self.update_field(fast=self.dragging)

    def fit_domain(self):
        self.plot.setRange(
            xRange=(-self.domain, self.domain),
            yRange=(-self.domain, self.domain),
            padding=0
        )

    def expand_domain(self, position):
        distance = np.max(np.abs(position))

        if not np.isfinite(distance):
            return False

        if distance <= self.domain * 0.85:
            return False

        self.domain = max(
            self.domain * 1.5,
            distance * 1.25
        )

        self.invalidate_cache()
        self.fit_domain()

        return True

    def on_plot_click(self, event):
        if not self.view.sceneBoundingRect().contains(
            event.scenePos()
        ):
            return

        point = self.view.mapSceneToView(
            event.scenePos()
        )

        x = point.x()
        y = point.y()

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.probe_point = np.array([x, y])
            self.update_probe()
            event.accept()
            return

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        if self.placing_particle:
            self.particle_x.setValue(x)
            self.particle_y.setValue(y)

            self.create_particle()

            event.accept()
            return

        index = self.find_charge(x, y)

        if index is not None:
            self.selected_charge = index
            self.sync_source_inputs()
            self.update_sources()

    def update_probe(self):
        if self.probe_point is None:
            self.probe_item.setData([], [])
            self.probe_vector.setData([], [])

            self.probe_label.setText(
                "Right-click to place probe"
            )

            return

        x, y = self.probe_point

        self.probe_item.setData([x], [y])

        try:
            field, magnitude, potential = (
                calculate_probe_values(
                    self.charges,
                    self.positions,
                    self.probe_point
                )
            )

            self.probe_label.setText(
                f"x = {x:.4f} m\n"
                f"y = {y:.4f} m\n\n"
                f"Ex = {field[0]:.4g} N/C\n"
                f"Ey = {field[1]:.4g} N/C\n"
                f"|E| = {magnitude:.4g} N/C\n"
                f"V = {potential:.4g} V"
            )

            if magnitude > 0:
                xmin, xmax, ymin, ymax = self.visible_bounds()

                span = min(
                    xmax - xmin, ymax - ymin
                )

                direction = field / magnitude

                end = (
                    self.probe_point
                    + direction * span * 0.07
                )

                arrows = self.create_arrow_segments(
                    end[None, :],
                    direction[None, :],
                    span * 0.015
                )

                coordinates = np.vstack([
                    self.probe_point,
                    end,
                    [np.nan, np.nan],
                    arrows
                ])

                self.probe_vector.setData(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    connect="finite"
                )
            else:
                self.probe_vector.setData([], [])

        except ValueError:
            self.probe_vector.setData([], [])

            self.probe_label.setText(
                "Undefined at point charge"
            )

    def clear_probe(self):
        self.probe_point = None
        self.update_probe()

    def place_particle(self):
        self.pause_particle(
            "Click the plot to place particle"
        )

        self.placing_particle = True

    def create_particle(self):
        position = np.array([
            self.particle_x.value(),
            self.particle_y.value()
        ])

        velocity = np.array([
            self.particle_vx.value(),
            self.particle_vy.value()
        ])

        active_positions = self.positions[
            self.charges != 0
        ]

        distances = np.linalg.norm(
            active_positions - position, axis=1
        )

        if np.any(distances < 0.08):
            self.set_status(
                "Particle too close to source charge"
            )
            return False

        self.particle = ParticleSimulator(
            position=position,
            velocity=velocity,
            particle_charge=self.particle_q.value() * 1e-9,
            particle_mass=self.particle_mass.value() * 1e-9,
            dt=0.001,
            method=self.integrator.currentText()
        )

        self.placing_particle = False

        self.start_button.setText("Start Simulation")

        if self.expand_domain(position):
            self.update_field()

        self.update_particle_graphics()
        self.update_particle_data()
        self.set_status("Ready")

        return True

    def pause_particle(self, status="Paused"):
        if self.particle is not None:
            self.particle.pause()

        self.start_button.setText("Start Simulation")
        self.set_status(status)

    def toggle_particle(self):
        if self.particle is None:
            if not self.create_particle():
                return

        if self.particle.running:
            self.pause_particle()
            self.update_particle_data()
        else:
            self.placing_particle = False

            self.particle.start()

            self.start_button.setText("Pause Simulation")

            self.set_status("Running")

            self.frame_count = 0
            self.last_fps_update = time.perf_counter()

    def reset_particle(self):
        if self.particle is None:
            return

        self.particle.reset()
        self.pause_particle("Ready")

        self.update_particle_graphics()
        self.update_particle_data()

    def change_integrator(self, method):
        if self.particle is not None:
            self.pause_particle(
                f"Integrator changed to {method}"
            )

            self.particle.method = method
            self.update_particle_data()

    def update_particle_graphics(self):
        if self.particle is None:
            self.particle_item.setData([], [])
            self.trajectory_item.setData([], [])
            return

        position = self.particle.get_position()

        self.particle_item.setData(
            [position[0]], [position[1]]
        )

        history = self.particle.trajectory

        stride = max(
            1, (len(history) + 2999) // 3000
        )

        trajectory = np.asarray(history[::stride])

        if (len(history) - 1) % stride:
            trajectory = np.vstack([
                trajectory, position
            ])

        self.trajectory_item.setData(
            trajectory[:, 0],
            trajectory[:, 1]
        )

    def update_particle_data(self):
        if self.particle is None:
            self.data_label.setText("No particle")
            return

        position = self.particle.get_position()
        velocity = self.particle.get_velocity()

        try:
            kinetic, potential, total = (
                self.particle.get_energy(
                    self.charges,
                    self.positions
                )
            )

            energy = (
                f"K = {kinetic:.3e} J\n"
                f"U = {potential:.3e} J\n"
                f"E = {total:.3e} J"
            )

        except ValueError:
            energy = "Energy undefined"

        self.data_label.setText(
            f"Method: {self.particle.method}\n\n"
            f"t  = {self.particle.time:.3f} s\n"
            f"x  = {position[0]:.4f} m\n"
            f"y  = {position[1]:.4f} m\n"
            f"vx = {velocity[0]:.4f} m/s\n"
            f"vy = {velocity[1]:.4f} m/s\n\n"
            f"{energy}"
        )

    def update_performance_label(self):
        self.performance_label.setText(
            f"Update FPS: {self.fps:.1f}\n"
            f"Physics: {self.physics_time:.2f} ms\n"
            f"Field update: {self.render_time:.2f} ms"
        )

    def update_simulation(self):
        now = time.perf_counter()

        if self.dragging:
            if (
                self.field_dirty
                and now - self.last_field_update >= 1 / 30
            ):
                self.update_field(fast=True)

            return

        if self.particle is None:
            return

        if not self.particle.running:
            return

        start = time.perf_counter()

        active_positions = self.positions[
            self.charges != 0
        ]

        for _ in range(10):
            position = self.particle.get_position()

            distances = np.linalg.norm(
                active_positions - position,
                axis=1
            )

            if np.any(distances < 0.08):
                self.pause_particle(
                    "Near point charge: stopped"
                )
                break

            previous_state = self.particle.state.copy()
            previous_time = self.particle.time

            previous_length = len(
                self.particle.trajectory
            )

            try:
                self.particle.step(
                    self.charges,
                    self.positions
                )

                if not np.all(
                    np.isfinite(self.particle.state)
                ):
                    raise FloatingPointError()

            except (ValueError, FloatingPointError):
                self.particle.state = previous_state
                self.particle.time = previous_time

                del self.particle.trajectory[
                    previous_length:
                ]

                self.pause_particle(
                    "Numerical singularity: stopped"
                )

                break

        self.physics_time = (
            time.perf_counter() - start
        ) * 1000

        if self.expand_domain(
            self.particle.get_position()
        ):
            self.update_field()

        self.update_particle_graphics()

        self.frame_count += 1

        elapsed = now - self.last_fps_update

        if elapsed >= 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = now

        if (
            now - self.last_text_update >= 0.1
            or not self.particle.running
        ):
            self.update_particle_data()
            self.update_performance_label()
            self.last_text_update = now

    def reset_all(self):
        self.pause_particle("Ready")

        self.charges = np.array([5e-9, -5e-9])

        self.positions = np.array([
            [-2.0, 0.0],
            [2.0, 0.0]
        ])

        self.selected_charge = 0
        self.domain = 5.0

        self.particle = None
        self.probe_point = None

        self.dragging = False
        self.placing_particle = False
        self.field_dirty = False

        self.fps = 0.0
        self.physics_time = 0.0

        self.particle_x.setValue(0)
        self.particle_y.setValue(2)

        self.particle_vx.setValue(0)
        self.particle_vy.setValue(1)

        self.particle_q.setValue(1)
        self.particle_mass.setValue(1)

        self.invalidate_cache()

        self.fit_domain()
        self.sync_source_inputs()

        self.update_sources()
        self.update_field()

        self.update_particle_graphics()
        self.update_particle_data()
        self.update_performance_label()


def run_interactive_simulator():
    app = pg.mkQApp("Electric Field Simulator")

    window = ElectricFieldSimulator()
    window.show()

    app.exec()


if __name__ == "__main__":
    run_interactive_simulator()