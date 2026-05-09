from src.error_model import LogicalErrorModel


# simulate trotter circuit for n*n grid of qubits and t trotter steps
def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    execution_logs: list[list[dict]],
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
    factory_state_fidelity = [0.0 for _ in range(n_factories)]
    fidelity_of_rz_injection = 1
    fidelity_of_rz_teleportaion = 1
    fidelity_of_rz_s = 1
    n_s = 0  # S in injection flow
    n_cnot = 0  # Clifford CNOTs
    n_cnot_teleportation = 0  # Injection CNOTs
    n_h = 0
    n_rz_seen = 0
    for log in execution_logs:
        for entry in log:
            operation = entry["operation"]
            factories = entry["factories"]
            values = entry["targets"]

            if operation == "H":
                targets = values if isinstance(values, list) else []
                n_h += len(targets)
                continue

            if operation == "CNOT" and isinstance(values, list):
                pairs = values
                n_cnot += len(pairs) if isinstance(pairs, list) else 0
                continue

            if operation == "Barrier":
                continue
            assert operation in [
                "SE",
                "SE_q",
                "CNOT",
                "Rz",
                "S",
                "TMR_fail",
                "RUS_success",
                "RUS_fail",
            ], f"Unexpected operation {operation} in execution log"
            if operation == "Rz":
                n_rz_seen += 1
                fac_list = factories if isinstance(factories, list) else []
                val_list = values if isinstance(values, list) else []
                for factory_id, value in zip(fac_list, val_list):
                    factory_state_fidelity[factory_id] = (
                        logical_error_model.get_rotation_fidelity(angle=value)
                    )
            elif operation == "S":
                n_s += len(values) if isinstance(values, (list, tuple)) else 1
                fidelity_of_rz_s *= logical_error_model.get_logical_fidelity("S")
            elif operation == "CNOT":
                fac_list = factories if isinstance(factories, list) else []
                val_list = values if isinstance(values, list) else []
                for factory_id, _value in zip(fac_list, val_list):
                    fidelity_of_rz_injection *= factory_state_fidelity[factory_id]
                    fidelity_of_rz_teleportaion *= (
                        logical_error_model.get_logical_fidelity("CNOT")
                    )
                n_cnot_teleportation += 1
    assert (
        n_rz_seen == n_rz_layer
    ), f"Expected {n_rz_layer} RZ rounds, but got {n_rz_seen}"
    fidelity_of_rz_teleportaion *= logical_error_model.get_logical_fidelity("CNOT") ** n_cnot_teleportation
    fidelity_cnot = logical_error_model.get_logical_fidelity("CNOT") ** n_cnot
    fidelity_1q = logical_error_model.get_logical_fidelity("H") ** n_h
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
        "n_cnot": n_cnot,
        "n_cnot_teleportation": n_cnot_teleportation,
        "n_h": n_h,
        "n_s": n_s,
    }
    return fidelity_profile
