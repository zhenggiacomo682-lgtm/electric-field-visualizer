import numpy as np
COULOMB_CONSTANT = 8.9875517923e9

def electric_field(q, charge_position, observation_point):
    displacement = observation_point - charge_position
    distance = np.linalg.norm(displacement)
    if distance == 0:
        raise ValueError("Electric field is undefined at the charge position.")
    
    field = COULOMB_CONSTANT * q * displacement / distance**3
    
    return field

def total_electric_field(charges, charge_positions, observation_point):
    total_field = np.zeros_like(observation_point, dtype=float)

    for i in range(len(charges)):
        field = electric_field(
            charges[i],
            charge_positions[i],
            observation_point
        )

        total_field += field

    return total_field