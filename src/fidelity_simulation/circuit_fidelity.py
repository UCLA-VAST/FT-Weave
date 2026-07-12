"""General circuit logical fidelity simulators (STAR and T-cultivation)."""

from __future__ import annotations

from src.circuit.gate_set import count_rz_layers
from src.circuit.rz_params import rz_target_angles
from src.error_model import LogicalErrorModel


def simulate_star_circuit_fidelity(
    *,
    n_factories: int,
    circuit: list[dict],
    execution_logs: list[list[dict]],
    logical_error_model: LogicalErrorModel,
    n_rz_expected: int | None = None,
) -> dict:
    """Estimate logical fidelity for a STAR-compiled Clifford+Rz circuit."""
    if n_rz_expected is None:
        n_rz_expected = count_rz_layers(circuit)

    factory_state_fidelity = [0.0 for _ in range(n_factories)]
    fidelity_of_rz_injection = 1.0
    fidelity_of_rz_s = 1.0
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
                targets = values if isinstance(values, list) else []
                n_se_q += len(targets)
                continue

            if operation == "CNOT":
                fac_list = factories if isinstance(factories, list) else []
                val_list = values if isinstance(values, list) else []
                if not fac_list:
                    n_cnot += len(val_list)
                else:
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

    if n_rz_expected > 0:
        assert (
            n_rz_seen >= n_rz_expected
        ), f"Expected at least {n_rz_expected} RZ rounds, but got {n_rz_seen}"

    cnot_fid = logical_error_model.get_logical_fidelity("CNOT")
    fidelity_of_rz_teleportaion = cnot_fid**n_cnot_teleportation
    fidelity_cnot = cnot_fid**n_cnot
    fidelity_1q = logical_error_model.get_logical_fidelity("H") ** n_h
    fidelity_idle = logical_error_model.get_logical_fidelity("I") ** n_se_q
    fidelity_of_rz_layer = (
        fidelity_of_rz_injection * fidelity_of_rz_teleportaion * fidelity_of_rz_s
    )
    fidelity = fidelity_of_rz_layer * fidelity_cnot * fidelity_1q * fidelity_idle
    return {
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
        "n_rz_seen": n_rz_seen,
    }


def simulate_t_cultivation_circuit_fidelity(
    *,
    n_factories: int,
    circuit: list[dict],
    execution_logs: list[list[dict]],
    logical_error_model: LogicalErrorModel,
    synthesis_epsilon: float | None = None,
    n_trotter_steps: int = 1,
) -> dict:
    """Estimate logical fidelity for a T-cultivation-compiled circuit."""
    n_cnot = 0
    n_cnot_teleportation = 0
    n_h_in_rz = 0
    n_s = 0
    n_t = 0
    n_se_q = 0

    for log in execution_logs:
        if not isinstance(log, list):
            continue
        for entry in log:
            factories = entry.get("factories")
            operation = entry.get("operation")
            targets = entry.get("targets")
            move_vecs = entry.get("move_vecs")

            if operation in (
                "move",
                "return_move",
                "Barrier",
                "RUS_success",
                "RUS_fail",
                "stage_2_success",
                "stage_2_fail",
                "X",
                "Z",
                "Y",
            ):
                continue
            if operation in ("SE", "SE_stage_1", "SE_stage_2"):
                continue

            if operation == "SE_q":
                tlist = targets if isinstance(targets, list) else []
                n_se_q += len(tlist)
                continue

            if operation == "CNOT":
                fac_list = (
                    list(factories)
                    if isinstance(factories, (list, tuple))
                    else ([factories] if factories is not None else [])
                )
                tgt_list = (
                    list(targets) if isinstance(targets, (list, tuple)) else []
                )
                if not fac_list:
                    if move_vecs is None:
                        n_cnot += len(tgt_list)
                else:
                    for factory_id, _qubit in zip(fac_list, tgt_list):
                        if not isinstance(factory_id, int):
                            continue
                        if not (0 <= factory_id < n_factories):
                            continue
                        n_cnot_teleportation += 1
                        n_t += 1
                continue

            if operation in ("T", "Tdg"):
                raise AssertionError(
                    "Direct T/Tdg gates are not supported in T-cultivation logs"
                )

            if operation == "S":
                n_s += 1
                continue

            if operation == "H":
                tlist = targets if isinstance(targets, list) else []
                n_h_in_rz += len(tlist) if tlist else 1
                continue

            raise ValueError(f"Unexpected operation in execution log: {operation}")

    n_cnot *= n_trotter_steps
    n_cnot_teleportation *= n_trotter_steps
    n_t *= n_trotter_steps
    n_h_in_rz *= n_trotter_steps
    n_s *= n_trotter_steps
    n_se_q *= n_trotter_steps

    n_rz_decomposition = (
        sum(
            len(rz_target_angles(inst))
            for inst in circuit
            if inst.get("gate") == "Rz"
        )
        * n_trotter_steps
    )

    cnot_fid = logical_error_model.get_logical_fidelity("CNOT")
    fidelity_cnot = cnot_fid**n_cnot
    fidelity_of_rz_teleportaion = cnot_fid**n_cnot_teleportation
    fidelity_of_rz_s = logical_error_model.get_logical_fidelity("S") ** n_s
    fidelity_h = logical_error_model.get_logical_fidelity("H") ** n_h_in_rz
    fidelity_idle = logical_error_model.get_logical_fidelity("I") ** n_se_q
    fidelity_of_t_gate = logical_error_model.get_logical_fidelity("T") ** n_t

    if synthesis_epsilon is None:
        fidelity_synthesis = 1.0
    else:
        eps = float(synthesis_epsilon)
        if eps < 0:
            raise ValueError(f"synthesis_epsilon must be non-negative, got {eps}")
        fidelity_synthesis = (1.0 - eps * eps) ** n_rz_decomposition

    fidelity_of_rz_h = fidelity_h
    fidelity_of_rz_layer = (
        fidelity_of_rz_teleportaion
        * fidelity_of_rz_s
        * fidelity_of_rz_h
        * fidelity_of_t_gate
    )
    fidelity = (
        fidelity_of_rz_layer * fidelity_cnot * fidelity_idle * fidelity_synthesis
    )

    return {
        "fidelity": fidelity,
        "fidelity_of_rz_layer": fidelity_of_rz_layer,
        "fidelity_of_rz_teleportaion": fidelity_of_rz_teleportaion,
        "fidelity_of_rz_s": fidelity_of_rz_s,
        "fidelity_of_rz_h": fidelity_of_rz_h,
        "fidelity_of_t_gate": fidelity_of_t_gate,
        "fidelity_cnot": fidelity_cnot,
        "fidelity_h": fidelity_h,
        "fidelity_idle": fidelity_idle,
        "fidelity_synthesis": fidelity_synthesis,
        "synthesis_epsilon": synthesis_epsilon,
        "n_cnot": n_cnot,
        "n_cnot_teleportation": n_cnot_teleportation,
        "n_h": n_h_in_rz,
        "n_s": n_s,
        "n_t": n_t,
        "n_se_q": n_se_q,
        "n_rz_decomposition": n_rz_decomposition,
    }
