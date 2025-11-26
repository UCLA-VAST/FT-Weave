from src.tfim import tfim_trotter_plaquette
from src.ds.architecture import Architecture
from src.ds.logical_grid import LogicalGridManager
from src.writer.writer import Writer
import json

# Example usage: 4x4 lattice, J=1.0, h=0.7, t=1.0, 2 trotter steps
if __name__ == "__main__":
    Lx, Ly = 4, 4
    J = 1.0
    h = 0.7
    t_total = 1.0
    n_steps = 2

    logic_qubit_locations = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 0, 2),
        (0, 1, 2),
        (0, 2, 2),
        (0, 3, 2),
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 3),
        (0, 3, 3),
    ]
    magic_state_locations = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 0, 2),
        (0, 1, 2),
        (0, 2, 2),
        (0, 3, 2),
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 3),
        (0, 3, 3),
    ]

    code_distance = 3
    arch_file = "hardware_spec/logical_architecture.json"
    with open(arch_file, "r") as file:
        arch_spec = json.load(file)

    arch = Architecture(arch_spec)

    logical_grid = LogicalGridManager(arch, code_distance=3)

    circuit_instructions = tfim_trotter_plaquette(
        Lx,
        Ly,
        J,
        h,
        t_total,
        n_steps,
        logic_qubit_locations,
        magic_state_locations,
        logical_grid,
    )

    n_q = len(logic_qubit_locations) * logical_grid.num_total_qubit
    qubit_locations = []
    for i, loc in enumerate(logic_qubit_locations):
        physical_qubit_indices, physical_qubit_locations = (
            logical_grid.physical_indices_location_for_logical_qubit(i, loc)
        )
        qubit_locations += physical_qubit_locations

    config_writer = config = {
        "architecture": arch,
        "n_q": n_q,
        "qubit_mapping": logic_qubit_locations,
    }
    writer = Writer(data=config_writer)
