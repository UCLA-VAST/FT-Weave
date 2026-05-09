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

    The execution log contains two distinct kinds of CNOT entries that we must
    keep separate when accounting for fidelity:

    1. Circuit-level Clifford CNOTs - logical CNOTs between two logical
       qubits emitted by the compiled TFIM circuit itself. These entries have
       an empty ``factories`` list and ``targets`` is a list of (control,
       target) qubit pairs (i.e. ``list[tuple[int, int]]``). They contribute
       to ``n_cnot`` / ``fidelity_cnot``.

    2. Teleportation / injection CNOTs - logical CNOTs between a magic-state
       factory and the logical qubit it injects into during the Rz round.
       These entries have a non-empty ``factories`` list and ``targets`` is a
       parallel list of bare qubit ids (i.e. ``list[int]``). They contribute
       to ``n_cnot_teleportation`` / ``fidelity_of_rz_teleportaion`` and they
       also fold the prepared factory state's fidelity into
       ``fidelity_of_rz_injection``.

    The discriminator is ``factories``: empty -> circuit CNOT, non-empty ->
    teleportation CNOT.
    """
    row, col = qubit_layout
    n_rz_layer = 9 * n_trotter_steps
    factory_state_fidelity = [0.0 for _ in range(n_factories)]
    fidelity_of_rz_injection = 1
    fidelity_of_rz_s = 1
    n_s = 0
    n_cnot = 0
    n_cnot_teleportation = 0
    n_h = 0
    n_se_q = 0
    n_rz_seen = 0
    for log in execution_logs:
        for entry in log:
            operation = entry["operation"]
            factories = entry["factories"]
            values = entry["targets"]

            # Movement / barrier entries do not directly contribute to logical
            # fidelity in this simple model (idle damage during transport is
            # captured separately at the script level if desired).
            if operation in ("Barrier", "move", "return_move"):
                continue
            assert operation in [
                "SE",
                "SE_q",
                "CNOT",
                "Rz",
                "S",
                "H",
                "TMR_fail",
                "RUS_success",
                "RUS_fail",
            ], f"Unexpected operation {operation} in execution log"

            if operation == "H":
                targets = values if isinstance(values, list) else []
                n_h += len(targets)
                continue

            if operation == "SE_q":
                # Logical-qubit syndrome-extraction events: each target qubit
                # contributes one idle-error round. Entries are coalesced (one
                # log entry per moment with multiple targets), so we count the
                # number of qubit-SE rounds = sum of |targets| across entries.
                targets = values if isinstance(values, list) else []
                n_se_q += len(targets)
                continue

            if operation == "CNOT":
                fac_list = factories if isinstance(factories, list) else []
                val_list = values if isinstance(values, list) else []
                if not fac_list:
                    # Empty factories => circuit-level Clifford CNOT between
                    # two logical qubits. ``targets`` is a list of
                    # (control, target) pairs, so each pair is one logical
                    # CNOT.
                    n_cnot += len(val_list)
                else:
                    # Non-empty factories => teleportation/injection CNOT.
                    # ``factories[i]`` is paired with ``targets[i]``; each
                    # such pair is one logical CNOT that consumes the
                    # corresponding factory's prepared Rz state.
                    for factory_id, _qubit in zip(fac_list, val_list):
                        fidelity_of_rz_injection *= factory_state_fidelity[factory_id]
                        n_cnot_teleportation += 1
                continue

            if operation == "Rz":
                n_rz_seen += 1
                fac_list = factories if isinstance(factories, list) else []
                val_list = values if isinstance(values, list) else []
                for factory_id, value in zip(fac_list, val_list):
                    factory_state_fidelity[factory_id] = (
                        logical_error_model.get_rotation_fidelity(angle=value)
                    )
                continue

            if operation == "S":
                n_s += len(values) if isinstance(values, (list, tuple)) else 1
                fidelity_of_rz_s *= logical_error_model.get_logical_fidelity("S")
                continue

    # Each Rz log entry corresponds to one TMR attempt; failed TMR rounds
    # retry, so we may see at least (and possibly more than) the nominal number
    # of Rz layers.
    assert (
        n_rz_seen >= n_rz_layer
    ), f"Expected at least {n_rz_layer} RZ rounds, but got {n_rz_seen}"

    cnot_fid = logical_error_model.get_logical_fidelity("CNOT")
    fidelity_of_rz_teleportaion = cnot_fid**n_cnot_teleportation
    fidelity_cnot = cnot_fid**n_cnot
    fidelity_1q = logical_error_model.get_logical_fidelity("H") ** n_h
    # Idle-error contribution: each qubit-SE round costs one factor of the
    # idle (identity) logical fidelity.
    fidelity_idle = logical_error_model.get_logical_fidelity("I") ** n_se_q
    fidelity_of_rz_layer = (
        fidelity_of_rz_injection * fidelity_of_rz_teleportaion * fidelity_of_rz_s
    )
    fidelity = fidelity_of_rz_layer * fidelity_cnot * fidelity_1q * fidelity_idle
    fidelity_profile = {
        "fidelity": fidelity,
        "fidelity_of_rz_layer": fidelity_of_rz_layer,
        "fidelity_of_rz_injection": fidelity_of_rz_injection,
        "fidelity_of_rz_teleportaion": fidelity_of_rz_teleportaion,
        "fidelity_of_rz_s": fidelity_of_rz_s,
        "fidelity_cnot": fidelity_cnot,
        "fidelity_1q": fidelity_1q,
        "fidelity_idle": fidelity_idle,
        "n_cnot": n_cnot,
        "n_cnot_teleportation": n_cnot_teleportation,
        "n_h": n_h,
        "n_s": n_s,
        "n_se_q": n_se_q,
    }
    return fidelity_profile
