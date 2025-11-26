from src.tfim import tfim_trotter_plaquette


# Example usage: 4x4 lattice, J=1.0, h=0.7, t=1.0, 2 trotter steps
if __name__ == "__main__":
    Lx, Ly = 4, 4
    J = 1.0
    h = 0.7
    t_total = 1.0
    n_steps = 2

    logic_qubit_locations = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    magic_state_locations = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]

    circuit_instructions = tfim_trotter_plaquette(
        Lx, Ly, J, h, t_total, n_steps, logic_qubit_locations, magic_state_locations
    )
