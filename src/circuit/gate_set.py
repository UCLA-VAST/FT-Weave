"""Gate-set classification for general circuit evaluation."""

from __future__ import annotations

from enum import Enum

CLIFFORD_GATES = frozenset(
    {"H", "S", "Sdg", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "I"}
)
RZ_GATES = frozenset({"Rz"})
T_GATES = frozenset({"T", "Tdg"})
SUPPORTED_GATES = CLIFFORD_GATES | RZ_GATES | T_GATES


class GateSetKind(str, Enum):
    CLIFFORD_RZ = "clifford_rz"
    CLIFFORD_T = "clifford_t"
    CLIFFORD_T_DECOMPOSED = "clifford_t_decomposed"
    PURE_CLIFFORD = "pure_clifford"
    UNSUPPORTED = "unsupported"


def is_clifford_gate(gate_name: str) -> bool:
    return gate_name in CLIFFORD_GATES


def is_non_clifford_gate(gate_name: str) -> bool:
    return gate_name in RZ_GATES or gate_name in T_GATES


def _unsupported_gates(circuit: list[dict]) -> list[str]:
    bad: list[str] = []
    for instr in circuit:
        gate = instr.get("gate", "")
        if gate not in SUPPORTED_GATES:
            bad.append(gate)
    return sorted(set(bad))


def classify_circuit(circuit: list[dict]) -> GateSetKind:
    """Classify a flat instruction list by native non-Clifford content."""
    unsupported = _unsupported_gates(circuit)
    if unsupported:
        return GateSetKind.UNSUPPORTED

    has_rz = any(instr.get("gate") == "Rz" for instr in circuit)
    has_t = any(instr.get("gate") in T_GATES for instr in circuit)

    if not has_rz and not has_t:
        return GateSetKind.PURE_CLIFFORD
    if has_t and not has_rz:
        return GateSetKind.CLIFFORD_T
    if has_rz and not has_t:
        return GateSetKind.CLIFFORD_RZ
    # Both Rz and bare T/Tdg present — treat as unsupported for auto mode.
    return GateSetKind.UNSUPPORTED


def count_rz_layers(circuit: list[dict]) -> int:
    """Count Rz layer instructions (each may target multiple qubits)."""
    return sum(1 for instr in circuit if instr.get("gate") == "Rz")


def validate_circuit_for_backend(
    circuit: list[dict],
    *,
    backend: str,
) -> GateSetKind:
    """Validate circuit and return the effective gate-set kind for *backend*.

    Raises:
        ValueError: unsupported gates, pure Clifford, or incompatible mix.
    """
    kind = classify_circuit(circuit)
    unsupported = _unsupported_gates(circuit)

    if unsupported:
        raise ValueError(
            f"Unsupported gate(s) in circuit: {', '.join(unsupported)}"
        )
    if kind == GateSetKind.PURE_CLIFFORD:
        raise ValueError(
            "Circuit contains only Clifford gates; nothing to inject via STAR or T-cultivation."
        )

    if backend == "star":
        if kind not in (GateSetKind.CLIFFORD_RZ,):
            if kind == GateSetKind.CLIFFORD_T:
                raise ValueError(
                    "STAR backend requires Clifford+RZ circuits (Rz gates, no bare T/Tdg)."
                )
            raise ValueError(f"Circuit not compatible with STAR backend: {kind.value}")
        return GateSetKind.CLIFFORD_RZ

    if backend == "t_cultivation":
        if kind == GateSetKind.CLIFFORD_T:
            return GateSetKind.CLIFFORD_T
        if kind == GateSetKind.CLIFFORD_RZ:
            return GateSetKind.CLIFFORD_T_DECOMPOSED
        raise ValueError(
            f"Circuit not compatible with T-cultivation backend: {kind.value}"
        )

    if backend == "auto":
        if kind == GateSetKind.CLIFFORD_RZ:
            return GateSetKind.CLIFFORD_RZ
        if kind == GateSetKind.CLIFFORD_T:
            return GateSetKind.CLIFFORD_T
        if kind == GateSetKind.CLIFFORD_T_DECOMPOSED and force_t_decompose:
            return GateSetKind.CLIFFORD_T_DECOMPOSED
        if kind == GateSetKind.UNSUPPORTED:
            has_rz = any(instr.get("gate") == "Rz" for instr in circuit)
            has_t = any(instr.get("gate") in T_GATES for instr in circuit)
            if has_rz and has_t:
                raise ValueError(
                    "Circuit mixes Rz and bare T/Tdg gates; use one backend explicitly."
                )
        raise ValueError(f"Cannot auto-select backend for circuit: {kind.value}")

    raise ValueError(f"Unknown backend: {backend}")
