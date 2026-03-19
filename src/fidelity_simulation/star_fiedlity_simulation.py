from src.error_model import LogicalErrorModel
import math


# simulate trotter circuit for n*n grid of qubits and t trotter steps
def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    qc_one_layer: list[dict],
    execution_logs: list[list[tuple]],
    logical_error_model: LogicalErrorModel,
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
    n_rz_layer = 9 * n_trotter_steps
    assert (
        len(execution_logs) == n_rz_layer
    ), f"Expected {n_rz_layer} layers of RZ gates, but got {len(execution_logs)}"

    factory_state_fidelity = [0.0 for _ in range(n_factories)]
    fidelity_of_rz_injection = 1
    fidelity_of_rz_teleportaion = 1
    fidelity_of_rz_s = 1
    for log in execution_logs:
        for entry in log:
            if len(entry) == 6:
                _, _, factories, operation, _, values = entry
                assert operation in [
                    "SE",
                    "CNOT",
                    "Rz",
                    "S",
                    "Barrier",
                    "TMR_fail",
                    "RUS_success",
                    "RUS_fail",
                ], f"Unexpected operation {operation} in execution log"
                if operation == "Rz":
                    # value is the angle of Rz gate, we assume the error is proportional to the angle
                    for factory_id, value in zip(factories, values):
                        factory_state_fidelity[factory_id] = (
                            logical_error_model.get_rotation_fidelity(angle=value)
                        )
                elif operation == "S":
                    # S gate has a fixed fidelity
                    # ! we don't need SE for S as S is in the middle of SE
                    fidelity_of_rz_s *= logical_error_model.get_logical_fidelity("S")
                elif operation == "CNOT":
                    # CNOT gate has a fixed fidelity
                    for factory_id, value in zip(factories, values):
                        fidelity_of_rz_injection *= factory_state_fidelity[factory_id]
                        fidelity_of_rz_teleportaion *= (
                            logical_error_model.get_logical_fidelity("CNOT")
                        )

    # count the clifford gates in one trotter step
    n_cnot_per_step = 0
    n_h_per_step = 0
    for instruction in qc_one_layer:
        if instruction["gate"] == "CNOT":
            n_cnot_per_step += len(instruction["targets"])
        elif instruction["gate"] == "H":
            n_h_per_step += len(instruction["targets"])

    fidelity_cnot = logical_error_model.get_logical_fidelity("CNOT") ** (
        n_cnot_per_step * n_trotter_steps
    )
    fidelity_1q = logical_error_model.get_logical_fidelity("H") ** (
        n_h_per_step * n_trotter_steps
    )
    fidelity_of_rz_layer = (
        fidelity_of_rz_injection * fidelity_of_rz_teleportaion * fidelity_of_rz_s
    )
    fidelity = fidelity_of_rz_layer * fidelity_cnot * fidelity_1q
    fidelity_profile = {
        "fidelity": fidelity,
        "fidelity_of_rz_layer": fidelity_of_rz_layer,
        "fidelity_of_rz_injection": fidelity_of_rz_injection,
        "fidelity_of_rz_teleportaion": fidelity_of_rz_teleportaion,
        "fidelity_of_rz_s": fidelity_of_rz_s,
        "fidelity_cnot": fidelity_cnot,
        "fidelity_1q": fidelity_1q,
        "n_cnot": n_cnot_per_step * n_trotter_steps,
        "n_h": n_h_per_step * n_trotter_steps,
    }
    return fidelity_profile
