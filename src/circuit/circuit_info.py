"""Helpers to infer circuit metadata."""

from __future__ import annotations


def count_qubits(circuit: list[dict]) -> int:
    """Return the number of qubits referenced in *circuit*."""
    if not circuit:
        return 0
    max_qubit = -1
    for instr in circuit:
        for q in _qubits_in_instruction(instr):
            max_qubit = max(max_qubit, q)
    return max_qubit + 1


def _qubits_in_instruction(instr: dict) -> list[int]:
    targets = instr.get("targets", [])
    qubits: list[int] = []
    for target in targets:
        if isinstance(target, int):
            qubits.append(target)
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            qubits.extend([int(target[0]), int(target[1])])
    return qubits
