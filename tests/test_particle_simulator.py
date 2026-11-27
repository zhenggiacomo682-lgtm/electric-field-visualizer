import numpy as np

from src.particle_simulator import ParticleSimulator

charges = np.array([-5e-9])

charge_positions = np.array([
    [0.0, 0.0]
])

simulator = ParticleSimulator(
    position=[2.0, 0.0],
    velocity=[0.0, 4.0],
    particle_charge=1e-9,
    particle_mass=1e-9,
    dt=0.001,
    method="RK4"
)

initial_state = simulator.state.copy()

simulator.step(charges, charge_positions)

np.testing.assert_allclose(
    simulator.state,
    initial_state
)

simulator.start()

for _ in range(1000):

    simulator.step(
        charges,
        charge_positions
    )

assert simulator.get_trajectory().shape == (1001, 2)

np.testing.assert_allclose(
    simulator.time,
    1.0
)

assert np.all(np.isfinite(simulator.state))

kinetic, potential, total = simulator.get_energy(
    charges,
    charge_positions
)

print("Simulation time:", simulator.time)
print("Final position:", simulator.get_position())
print("Final velocity:", simulator.get_velocity())
print("Kinetic energy:", kinetic)
print("Potential energy:", potential)
print("Total energy:", total)

simulator.pause()

paused_state = simulator.state.copy()

simulator.step(charges, charge_positions)

np.testing.assert_allclose(
    simulator.state,
    paused_state
)

simulator.reset()

np.testing.assert_allclose(
    simulator.state,
    initial_state
)

assert simulator.time == 0.0
assert simulator.running is False
assert simulator.get_trajectory().shape == (1, 2)

print("Particle simulator test passed successfully.")