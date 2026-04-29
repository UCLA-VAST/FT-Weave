"""Logical fidelity estimation for T-cultivation execution of TFIM layers.

Mirrors ``star_fiedlity_simulation.py``: ``execution_logs`` is a **list of logs**
(one per Rz round from ``generate_one_layer_2d_tfim_circuit_t_cultivation``). The
simulator loops ``for log in execution_logs: for entry in log: ...``.

The Clifford contribution is still derived from ``qc_one_layer`` and
``n_trotter_steps``, as in the STAR simulator.
"""

from __future__ import annotations

from typing import Any

from src.error_model import LogicalErrorModel


def _unpack_execution_entry(entry: tuple) -> tuple[Any, str, Any, Any]:
    """Normalize 5-, 6-, or 7-tuple execution log rows."""
    n = len(entry)
    if n == 7:
        _start, _end, factories, operation, aod, targets, _move_vecs = entry
    elif n == 6:
        _start, _end, factories, operation, aod, targets = entry
    elif n == 5:
        _start, _end, factories, operation = entry[:4]
        aod, targets = None, None
    else:
        raise ValueError(f"Unexpected execution log entry length: {n}")
    return factories, operation, aod, targets


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
    Estimate logical fidelity for TFIM evolution under T-cultivation.

    H=-J sum_{i,j} Z_iZ_j - h sum_i X_i. One Trotter step uses the same logical
    layer as STAR (CNOT / Rz / H in ``qc_one_layer``); Rz are executed as T/Tdg
    via RUS in the simulator.

    Args:
        n_qubits: Number of logical qubits.
        n_factories: Number of T factories (used to size factory state table).
        qubit_layout: (rows, cols) layout (same convention as elsewhere).
        n_trotter_steps: Number of Trotter steps; scales Clifford counts.
        qc_one_layer: One logical TFIM layer (``logical=True`` from
            ``generate_one_layer_2d_tfim_circuit_cz``).
        execution_logs: Per-Rz round logs from ``t_cultivation_execution``, same shape
            as STAR’s ``rz_logs`` (one inner list per ``Rz`` instruction in order).
        logical_error_model: Surface-code logical error model. For T-cultivation,
            call ``logical_error_model.integrate_t_cultivation_fidelity_target(p)``
            so ``get_logical_fidelity('T')`` uses logical T fidelity ``1 - p`` (default).

    Returns:
        Fidelity profile dict aligned with STAR naming where possible.
    """
    _row, _col = qubit_layout

    n_rz_instructions = sum(1 for inst in qc_one_layer if inst.get("gate") == "Rz")
    if n_rz_instructions != len(execution_logs):
        raise ValueError(
            f"Expected {n_rz_instructions} Rz round logs (one per Rz in qc_one_layer), "
            f"got {len(execution_logs)}"
        )

    fidelity_of_rz_teleportaion = 1.0
    fidelity_of_rz_s = 1.0
    fidelity_of_rz_h = 1.0
    fidelity_of_t_gate = 1.0

    n_cnot = 0
    n_s = 0
    for log in execution_logs:
        for entry in log:
            factories, operation, _aod, targets = _unpack_execution_entry(entry)

            if operation in ("move", "return_move", "Barrier"):
                continue

            if operation in ("SE_stage_1", "SE_stage_2"):
                # Factory prep is not part of the STAR analytic model; omit here (extend if needed).
                continue

            if operation == "CNOT":
                # RUS teleportation injection (same structure as STAR).
                if not isinstance(factories, (list, tuple)):
                    factory_list = [factories] if factories is not None else []
                else:
                    factory_list = list(factories)
                if not isinstance(targets, (list, tuple)):
                    qubit_list = [targets] if targets is not None else []
                else:
                    qubit_list = list(targets)
                for factory_id, _qubit in zip(factory_list, qubit_list):
                    if not isinstance(factory_id, int):
                        continue
                    if not (0 <= factory_id < n_factories):
                        continue
                    fidelity_of_rz_teleportaion *= (
                        logical_error_model.get_logical_fidelity("CNOT")
                    )
                    fidelity_of_t_gate *= logical_error_model.get_logical_fidelity("T")
                    n_cnot += 1
            elif operation in ("T", "Tdg"):
                fidelity_of_t_gate *= logical_error_model.get_logical_fidelity("T")
            elif operation == "S":
                fidelity_of_rz_s *= logical_error_model.get_logical_fidelity("S")
                n_s += 1
            elif operation == "H":
                fidelity_of_rz_h *= logical_error_model.get_logical_fidelity("H")
            elif operation in ("RUS_success", "RUS_fail"):
                # Outcomes are recorded separately; injection noise is tied to CNOT above.
                continue
            else:
                raise ValueError(f"Unexpected operation in execution log: {operation}")

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
    fidelity_h = logical_error_model.get_logical_fidelity("H") ** (
        n_h_per_step * n_trotter_steps
    )

    fidelity_of_rz_layer = (
        fidelity_of_rz_teleportaion
        * fidelity_of_rz_s
        * fidelity_of_rz_h
        * fidelity_of_t_gate
    )
    fidelity = fidelity_of_rz_layer * fidelity_cnot * fidelity_h

    return {
        "fidelity": fidelity,
        "fidelity_of_rz_layer": fidelity_of_rz_layer,
        "fidelity_of_rz_teleportaion": fidelity_of_rz_teleportaion,
        "fidelity_of_rz_s": fidelity_of_rz_s,
        "fidelity_of_rz_h": fidelity_of_rz_h,
        "fidelity_of_t_gate": fidelity_of_t_gate,
        "fidelity_cnot": fidelity_cnot,
        "fidelity_h": fidelity_h,
        "n_cnot": n_cnot + n_cnot_per_step * n_trotter_steps,
        "n_h": n_h_per_step * n_trotter_steps,
        "n_s": n_s,
    }
