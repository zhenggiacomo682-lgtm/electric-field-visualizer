import numpy as np

from matplotlib.colors import Normalize
from src.field_solver import calculate_field_grid_vectorized


class DynamicRenderer:

    def __init__(self, ax, domain):

        self.ax = ax
        self.domain = domain

        self.quiver = None
        self.charge_markers = None
        self.selection_marker = None
        self.labels = []

        self.X = None
        self.Y = None

    def calculate_vectors(self, charges, positions):

        active = charges != 0

        if not np.any(active):

            empty = np.ma.masked_all(self.X.shape)

            return empty, empty, empty

        Ex, Ey = calculate_field_grid_vectorized(
            charges[active],
            positions[active],
            self.X,
            self.Y
        )

        magnitude = np.hypot(Ex, Ey)

        valid = (
            np.isfinite(magnitude)
            & (magnitude > 0)
        )

        U = np.divide(
            Ex,
            magnitude,
            out=np.zeros_like(Ex),
            where=valid
        )

        V = np.divide(
            Ey,
            magnitude,
            out=np.zeros_like(Ey),
            where=valid
        )

        C = np.log1p(magnitude)

        return (
            np.ma.masked_where(~valid, U),
            np.ma.masked_where(~valid, V),
            np.ma.masked_where(~valid, C)
        )

    def get_colors(self, charges):

        return [
            "#ff4757" if q > 0
            else "#38bdf8" if q < 0
            else "#888888"
            for q in charges
        ]

    def initialize(
        self,
        charges,
        positions,
        selected_charge
    ):

        limit = self.domain.limit

        coordinates = np.linspace(
            -limit,
            limit,
            17
        )

        self.X, self.Y = np.meshgrid(
            coordinates,
            coordinates
        )

        U, V, C = self.calculate_vectors(
            charges,
            positions
        )

        self.ax.clear()

        self.ax.set_facecolor("#0d1117")

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        self.ax.set_aspect("equal")

        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")

        self.ax.set_title(
            "Electric Field | Interactive Preview",
            fontsize=14,
            fontweight="bold"
        )

        self.ax.grid(alpha=0.08)

        self.quiver = self.ax.quiver(
            self.X,
            self.Y,
            U,
            V,
            C,
            cmap="turbo",
            norm=Normalize(0, 5),
            angles="xy",
            scale_units="xy",
            scale=12 / limit,
            width=0.003,
            pivot="middle",
            zorder=2
        )

        self.charge_markers = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=100,
            c=self.get_colors(charges),
            edgecolors="white",
            zorder=5
        )

        self.selection_marker = self.ax.scatter(
            [],
            [],
            s=250,
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.5,
            zorder=6
        )

        self.labels = []

        for q, position, color in zip(
            charges,
            positions,
            self.get_colors(charges)
        ):

            label = self.ax.annotate(
                f"{q * 1e9:+.2f} nC",
                position,
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold",
                clip_on=True,
                zorder=7
            )

            self.labels.append(label)

        self.update_selection(
            positions,
            selected_charge
        )

    def update_selection(
        self,
        positions,
        selected_charge
    ):

        if selected_charge is None:

            self.selection_marker.set_offsets(
                np.empty((0, 2))
            )

        else:

            self.selection_marker.set_offsets(
                positions[selected_charge].reshape(1, 2)
            )

    def update_charges(
        self,
        charges,
        positions,
        selected_charge
    ):

        colors = self.get_colors(charges)

        self.charge_markers.set_offsets(
            positions
        )

        self.charge_markers.set_facecolors(
            colors
        )

        self.update_selection(
            positions,
            selected_charge
        )

        for label, q, position, color in zip(
            self.labels,
            charges,
            positions,
            colors
        ):

            label.xy = tuple(position)

            label.set_text(
                f"{q * 1e9:+.2f} nC"
            )

            label.set_color(color)

    def update_field(self, charges, positions):

        U, V, C = self.calculate_vectors(
            charges,
            positions
        )

        self.quiver.set_UVC(U, V, C)

    def update(
        self,
        charges,
        positions,
        selected_charge
    ):

        self.update_charges(
            charges,
            positions,
            selected_charge
        )

        self.update_field(
            charges,
            positions
        )

    def get_charge_artists(self):

        return [
            self.charge_markers,
            self.selection_marker,
            *self.labels
        ]

    def get_artists(self):

        return [
            self.quiver,
            *self.get_charge_artists()
        ]