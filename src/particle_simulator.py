import numpy as np

from src.particle_dynamics import (
    simulate_euler,
    simulate_rk4,
    simulate_verlet
)

from src.energy import calculate_particle_energy


class ParticleSimulator:

    def __init__(
        self,
        position,
        velocity,
        particle_charge,
        particle_mass,
        dt=0.001,
        method="RK4"
    ):

        if particle_mass <= 0:
            raise ValueError("Particle mass must be positive")

        if dt <= 0:
            raise ValueError("Time step must be positive")

        self.initial_state = np.array([
            position[0],
            position[1],
            velocity[0],
            velocity[1]
        ], dtype=float)

        self.state = self.initial_state.copy()

        self.particle_charge = particle_charge
        self.particle_mass = particle_mass
        self.dt = dt
        self.method = method

        self.time = 0.0
        self.running = False

        self.trajectory = [
            self.state[:2].copy()
        ]

    def start(self):

        self.running = True

    def pause(self):

        self.running = False

    def reset(self):

        self.state = self.initial_state.copy()

        self.time = 0.0
        self.running = False

        self.trajectory = [
            self.state[:2].copy()
        ]

    def step(self, charges, charge_positions):

        if not self.running:
            return

        methods = {
            "Euler": simulate_euler,
            "RK4": simulate_rk4,
            "Verlet": simulate_verlet
        }

        if self.method not in methods:
            raise ValueError("Unknown integration method")

        solver = methods[self.method]

        states = solver(
            self.state,
            self.dt,
            1,
            self.particle_charge,
            self.particle_mass,
            charges,
            charge_positions
        )

        self.state = states[-1].copy()

        self.time += self.dt

        self.trajectory.append(
            self.state[:2].copy()
        )

    def get_position(self):

        return self.state[:2].copy()

    def get_velocity(self):

        return self.state[2:].copy()

    def get_trajectory(self):

        return np.asarray(self.trajectory)

    def get_energy(self, charges, charge_positions):

        return calculate_particle_energy(
            self.state,
            self.particle_charge,
            self.particle_mass,
            charges,
            charge_positions
        )