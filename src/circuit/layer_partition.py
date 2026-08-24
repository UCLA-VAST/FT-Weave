"""Partition flat circuits into Clifford / non-Clifford layer instructions."""

from __future__ import annotations

from src.circuit.gate_set import is_clifford_gate, is_non_clifford_gate
from src.circuit.rz_params import (
    combine_rz_angles,
    rz_params_from_angles,
    rz_target_angles,
)


def _append_or_merge_rz_layer(layers: list[dict], targets: list, params: dict) -> None:
    """Merge consecutive Rz gates into a single layer between Clifford blocks."""
    gate_angles = rz_target_angles({"gate": "Rz", "targets": targets, "params": params})

    if layers and layers[-1]["gate"] == "Rz":
        prev = layers[-1]
        prev_angles = rz_target_angles(prev)
        merged_angles = combine_rz_angles(prev_angles, gate_angles)
        prev["targets"] = list(merged_angles.keys())
        prev["params"] = rz_params_from_angles(merged_angles)
        return

    layers.append(
        {
            "gate": "Rz",
            "targets": list(gate_angles.keys()),
            "params": rz_params_from_angles(gate_angles),
        }
    )


def partition_into_layers(flat_circuit: list[dict]) -> list[dict]:
    """Partition a flat gate list into layer instructions.

    Consecutive Rz gates (even with different angles) merge into one Rz layer
    between Clifford blocks. Repeated Rz on the same qubit sums angles.
    Clifford gates are emitted one instruction per gate to preserve QASM order.
    Non-Clifford T/Tdg gates each remain their own layer.
    """
    if not flat_circuit:
        return []

    layers: list[dict] = []
    for instr in flat_circuit:
        gate = instr["gate"]
        targets = instr.get("targets", [])
        params = instr.get("params", {})

        if is_non_clifford_gate(gate):
            if gate == "Rz":
                _append_or_merge_rz_layer(layers, targets, params)
            else:
                layers.append(
                    {"gate": gate, "targets": list(targets), "params": dict(params)}
                )
        elif is_clifford_gate(gate):
            layers.append(
                {"gate": gate, "targets": list(targets), "params": dict(params)}
            )
        else:
            raise ValueError(f"Unknown gate in partition: {gate}")

    return layers
