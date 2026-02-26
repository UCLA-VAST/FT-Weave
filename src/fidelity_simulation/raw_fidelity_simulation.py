from src.error_model import PhysicalErrorModel
import math


# simulate trotter circuit for n*n grid of qubits and t trotter steps
def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    qc_one_layer: list[dict],
    physical_error_model: PhysicalErrorModel,
    site_seperation: float = 10 * 1e-6,  # m
    trap_seperation: float = 2 * 1e-6,  # m
    a: float = 5500,  # m/s^2
    gate_duration_1q: float = 8e-6,  # seconds
    gate_duration_cz: float = 360 * 1e-9,  # seconds
    atom_transfer_duration: float = 15e-6,  # seconds
) -> dict:
    """
    H=-J sum_{i,j} Z_iZ_j - h sum_i X_i
    One trotter step consists of:
        [ZZ horizontal even]
        [ZZ horizontal odd]
        [ZZ vertical even]
        [ZZ vertical odd]
        [RX on all qubits]
    """
    row, col = qubit_layout

    # process the circuit to count the number of gates and movements, and compute the idle time for each qubit
    circuit_duration_per_step = 0.0
    qubit_busy_time_per_step = [0.0 for _ in range(n_qubits)]
    n_cz_per_step = 0
    n_cz_spec_per_step = 0
    n_cz_layer_per_step = 0
    n_1q_per_step = 0
    for instruction in qc_one_layer:
        if instruction["gate"] == "CZ":
            circuit_duration_per_step += gate_duration_cz
            n_cz_per_step += len(instruction["targets"])
            n_cz_spec_per_step += n_qubits - len(instruction["targets"]) * 2
            n_cz_layer_per_step += 1
            for q0, q1 in instruction["targets"]:
                qubit_busy_time_per_step[q0] += gate_duration_cz
                qubit_busy_time_per_step[q1] += gate_duration_cz
        elif instruction["gate"] in ["H", "Rx"]:
            # count h as one 1q gate
            circuit_duration_per_step += gate_duration_1q
            n_1q_per_step += len(instruction["targets"])
            for q in instruction["targets"]:
                qubit_busy_time_per_step[q] += gate_duration_1q

    # movement count
    n_movement_per_step = (row + row // 2) * col + (col + col // 2) * row

    # compute idle time for each qubit in one trotter step
    # movement assumption: single qubit gates does not require movement
    # idle time includes movement time, atom transfer time and waiting time for gates to finish
    cross_trap_movement_duration = math.sqrt(trap_seperation / 2 / a) * 2  # s
    cross_site_movement_duration = math.sqrt(site_seperation / 2 / a) * 2  # s
    idle_time_per_step = (
        cross_site_movement_duration  # move to first ZZ layer
        + cross_site_movement_duration  # move to second ZZ layer
        + cross_site_movement_duration  # return to init configuration
        + cross_trap_movement_duration
        + cross_site_movement_duration  # move to third ZZ layer
        + cross_site_movement_duration  # move to fourth ZZ layer
        + cross_site_movement_duration  # return to init configuration
        + atom_transfer_duration * 2 * 6  # pick up and drop for 5 movement layers
    )

    # add atom transfer time for movement layers
    circuit_duration_per_step += idle_time_per_step

    fidelity_init = physical_error_model.get_fidelity("init") ** n_qubits
    fidelity_measurement = physical_error_model.get_fidelity("measurement") ** n_qubits
    fidelity_cz = physical_error_model.get_fidelity("cz_loss") ** (
        n_cz_layer_per_step * n_trotter_steps * n_qubits
    )  # cz loss is applied to all qubits in the layer
    fidelity_cz *= physical_error_model.get_fidelity("cz") ** (
        n_cz_per_step * n_trotter_steps
    )
    fidelity_cz *= physical_error_model.get_fidelity("cz_spec") ** (
        n_cz_spec_per_step * n_trotter_steps
    )
    fidelity_1q = physical_error_model.get_fidelity("1q") ** (
        n_1q_per_step * n_trotter_steps
    )
    fidelity_move = physical_error_model.get_fidelity("move") ** (
        n_movement_per_step * n_trotter_steps
    )
    fidelity_move *= physical_error_model.get_fidelity("move_loss") ** (
        n_movement_per_step * n_trotter_steps
    )
    # idle time error
    fidelity_idle = 1.0
    for q in range(n_qubits):
        total_idle_time = (
            circuit_duration_per_step - qubit_busy_time_per_step[q]
        ) * n_trotter_steps
        # compute number of idle error events based on total idle time and coherence time
        n_idle_time = total_idle_time / physical_error_model.get_coherence_time()
        fidelity_idle *= math.exp(-n_idle_time)

    fidelity = (
        fidelity_init
        * fidelity_measurement
        * fidelity_cz
        * fidelity_1q
        * fidelity_move
        * fidelity_idle
    )
    fidelity_profile = {
        "total_duration": circuit_duration_per_step * n_trotter_steps,
        "fidelity": fidelity,
        "fidelity_init": fidelity_init,
        "fidelity_measurement": fidelity_measurement,
        "fidelity_cz": fidelity_cz,
        "fidelity_1q": fidelity_1q,
        "fidelity_move": fidelity_move,
        "fidelity_idle": fidelity_idle,
        "n_cz": n_cz_per_step * n_trotter_steps,
        "n_cz_spec": n_cz_spec_per_step * n_trotter_steps,
        "n_1q": n_1q_per_step * n_trotter_steps,
        "n_movement": n_movement_per_step * n_trotter_steps,
    }
    return fidelity_profile
