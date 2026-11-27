import numpy as np

from src.grid import create_grid


class DomainManager:

    def __init__(
        self,
        limit=5.0,
        fast_resolution=60,
        full_resolution=200
    ):

        self.fast_resolution = fast_resolution
        self.full_resolution = full_resolution

        self._limit = float(limit)
        self.initial_limit = float(limit)

        self.rebuild_grids()

    @property
    def limit(self):

        return self._limit

    @limit.setter
    def limit(self, value):

        value = float(value)

        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid domain limit")

        if value != self._limit:

            self._limit = value
            self.rebuild_grids()

    def rebuild_grids(self):

        self.X_fast, self.Y_fast = create_grid(
            -self.limit,
            self.limit,
            -self.limit,
            self.limit,
            self.fast_resolution
        )

        self.X_full, self.Y_full = create_grid(
            -self.limit,
            self.limit,
            -self.limit,
            self.limit,
            self.full_resolution
        )

    def get_grid(self, fast=False):

        if fast:
            return self.X_fast, self.Y_fast

        return self.X_full, self.Y_full

    def expand_if_needed(self, position):

        distance = np.max(np.abs(position))

        if not np.isfinite(distance):
            return False

        if distance <= self.limit * 0.85:
            return False

        self.limit = max(
            self.limit * 1.5,
            distance * 1.25
        )

        return True

    def reset(self):

        self.limit = self.initial_limit