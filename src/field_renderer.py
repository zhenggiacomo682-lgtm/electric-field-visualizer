import numpy as np

from matplotlib.colors import Normalize

from src.field_solver import (
    calculate_field_grid_vectorized,
    calculate_potential_grid
)


class FieldRenderer:

    def __init__(self, ax, domain):

        self.ax = ax
        self.domain = domain

    def draw_electric_field(
        self,
        charges,
        positions,
        X,
        Y,
        density,
        linewidth,
        fast=False
    ):

        if len(charges) == 0:
            return

        Ex, Ey = calculate_field_grid_vectorized(
            charges,
            positions,
            X,
            Y
        )

        magnitude = np.hypot(Ex, Ey)

        log_magnitude = np.log1p(magnitude)

        finite_values = log_magnitude[
            np.isfinite(log_magnitude)
        ]

        if not finite_values.size:
            return

        if np.max(finite_values) <= 0:
            return

        upper_limit = max(
            np.percentile(finite_values, 95),
            1e-12
        )

        norm = Normalize(
            vmin=0,
            vmax=upper_limit
        )

        if fast:

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

            self.ax.quiver(
                X,
                Y,
                np.ma.masked_where(~valid, U),
                np.ma.masked_where(~valid, V),
                np.ma.masked_where(
                    ~valid,
                    log_magnitude
                ),
                cmap="turbo",
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=22 / self.domain.limit,
                width=0.0025,
                pivot="middle",
                alpha=0.9
            )

        else:

            self.ax.streamplot(
                X,
                Y,
                np.ma.masked_invalid(Ex),
                np.ma.masked_invalid(Ey),
                color=np.ma.masked_invalid(
                    log_magnitude
                ),
                cmap="turbo",
                norm=norm,
                density=density,
                linewidth=linewidth,
                arrowsize=0.9
            )

    def draw_equipotential(
        self,
        charges,
        positions,
        X,
        Y,
        current_view,
        number_of_levels,
        fast
    ):

        if len(charges) == 0:
            return

        V = calculate_potential_grid(
            charges,
            positions,
            X,
            Y
        )

        finite_values = V[
            np.isfinite(V)
        ]

        if not finite_values.size:
            return

        limit = np.percentile(
            np.abs(finite_values),
            95
        )

        if limit <= 0:
            return

        levels = np.linspace(
            -limit,
            limit,
            number_of_levels
        )

        if current_view == "Equipotential":

            self.ax.contourf(
                X,
                Y,
                np.ma.masked_invalid(V),
                levels=levels,
                cmap="coolwarm",
                extend="both",
                alpha=0.85
            )

        contours = self.ax.contour(
            X,
            Y,
            np.ma.masked_invalid(V),
            levels=levels,
            colors="white",
            linewidths=0.7,
            linestyles="dashed",
            alpha=0.45
        )

        if not fast:

            self.ax.clabel(
                contours,
                inline=True,
                fontsize=7,
                fmt="%.1f V"
            )

    def draw_charges(
        self,
        charges,
        positions,
        selected_charge
    ):

        for i, (q, position) in enumerate(
            zip(charges, positions)
        ):

            if q > 0:

                color = "#ff4757"

            elif q < 0:

                color = "#38bdf8"

            else:

                color = "#888888"

            self.ax.scatter(
                position[0],
                position[1],
                s=100,
                color=color,
                edgecolors="white",
                zorder=5
            )

            if i == selected_charge:

                self.ax.scatter(
                    position[0],
                    position[1],
                    s=250,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=1.5,
                    zorder=6
                )

            self.ax.annotate(
                f"{q * 1e9:+.2f} nC",
                position,
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold"
            )

    def configure_axes(
        self,
        current_view,
        number_of_charges
    ):

        limit = self.domain.limit

        self.ax.set_xlim(
            -limit,
            limit
        )

        self.ax.set_ylim(
            -limit,
            limit
        )

        self.ax.set_aspect("equal")

        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")

        self.ax.set_title(
            f"{current_view} | {number_of_charges} charges",
            fontsize=14,
            fontweight="bold"
        )

        self.ax.grid(alpha=0.08)

    def render(
        self,
        charges,
        positions,
        selected_charge,
        current_view,
        fast=False
    ):

        X, Y = self.domain.get_grid(fast)

        if fast:

            X = X[::2, ::2]
            Y = Y[::2, ::2]

            density = 0.6
            linewidth = 0.8
            number_of_levels = 7

        else:

            density = 1.5
            linewidth = 1.0
            number_of_levels = 13

        self.ax.clear()

        self.ax.set_facecolor("#0d1117")

        active = charges != 0

        active_charges = charges[active]

        active_positions = positions[active]

        if current_view in [
            "Electric Field",
            "Combined"
        ]:

            self.draw_electric_field(
                active_charges,
                active_positions,
                X,
                Y,
                density,
                linewidth,
                fast
            )

        if current_view in [
            "Equipotential",
            "Combined"
        ]:

            self.draw_equipotential(
                active_charges,
                active_positions,
                X,
                Y,
                current_view,
                number_of_levels,
                fast
            )

        self.draw_charges(
            charges,
            positions,
            selected_charge
        )

        self.configure_axes(
            current_view,
            len(charges)
        )