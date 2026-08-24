"""Load OpenQASM circuits into the internal dict-based instruction IR."""

from __future__ import annotations

import math
from pathlib import Path

from qiskit import QuantumCircuit

# Internal gate name normalization (lowercase Qiskit name -> IR name)
_GATE_NAME_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "Sdg",
    "t": "T",
    "tdg": "Tdg",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
    "id": "I",
    "rz": "Rz",
    "p": "Rz",
    "u1": "Rz",
}

# One-qubit gates decomposed to H + Rz sequences before IR emission.
_DECOMPOSE_TO_RZ = frozenset({"rx", "ry", "u", "u2", "u3"})


def _normalize_gate_name(qiskit_name: str) -> str:
    key = qiskit_name.lower()
    if key in _GATE_NAME_MAP:
        return _GATE_NAME_MAP[key]
    raise ValueError(f"Unsupported QASM gate: {qiskit_name}")


def _qubit_indices(qc: QuantumCircuit, qubits) -> list[int]:
    indices: list[int] = []
    for q in qubits:
        if hasattr(qc, "find_bit"):
            indices.append(qc.find_bit(q).index)
        elif hasattr(qc, "find_bits"):
            indices.append(qc.find_bits([q])[0][0])
        else:
            indices.append(q._index)
    return indices


def _decompose_one_qubit_to_rz(qc: QuantumCircuit, circuit_instruction) -> list[dict]:
    """Decompose arbitrary one-qubit gates into H/S/Rz IR instructions."""
    op = circuit_instruction.operation
    qubit = _qubit_indices(qc, circuit_instruction.qubits)[0]
    name = op.name.lower()

    if name in ("rx", "ry"):
        angle = float(op.params[0])
        # Rx(theta) = H Rz(theta) H, Ry(theta) = Sdg H Rz(theta) H S
        if name == "rx":
            return [
                {"gate": "H", "targets": [qubit], "params": {}},
                {"gate": "Rz", "targets": [qubit], "params": {"theta": angle}},
                {"gate": "H", "targets": [qubit], "params": {}},
            ]
        return [
            {"gate": "Sdg", "targets": [qubit], "params": {}},
            {"gate": "H", "targets": [qubit], "params": {}},
            {"gate": "Rz", "targets": [qubit], "params": {"theta": angle}},
            {"gate": "H", "targets": [qubit], "params": {}},
            {"gate": "S", "targets": [qubit], "params": {}},
        ]

    # u, u2, u3, u1, p — use Qiskit to decompose to rz + clifford
    mini = QuantumCircuit(1)
    mini.append(op, [0])
    decomposed = mini.decompose().decompose()
    instructions: list[dict] = []
    for dec_inst in decomposed.data:
        dec_name = dec_inst.operation.name.lower()
        dec_qubit = _qubit_indices(mini, dec_inst.qubits)[0]
        if dec_name in _DECOMPOSE_TO_RZ or dec_name in ("u1", "p"):
            theta = (
                float(dec_inst.operation.params[0])
                if dec_inst.operation.params
                else 0.0
            )
            if dec_name in ("p", "u1"):
                instructions.append(
                    {"gate": "Rz", "targets": [dec_qubit], "params": {"theta": theta}}
                )
            else:
                sub = _decompose_one_qubit_to_rz(mini, dec_inst)
                for sub_inst in sub:
                    sub_inst["targets"] = [qubit]
                instructions.extend(sub)
        elif dec_name in _GATE_NAME_MAP:
            gate = _normalize_gate_name(dec_name)
            params: dict = {}
            if gate == "Rz" and dec_inst.operation.params:
                params["theta"] = float(dec_inst.operation.params[0])
            instructions.append({"gate": gate, "targets": [qubit], "params": params})
        else:
            raise ValueError(f"Cannot decompose gate {dec_name} from {name}")
    return instructions


def _instruction_to_ir(qc: QuantumCircuit, circuit_instruction) -> list[dict]:
    op = circuit_instruction.operation
    name = op.name.lower()

    if name in _DECOMPOSE_TO_RZ:
        return _decompose_one_qubit_to_rz(qc, circuit_instruction)

    qubits = _qubit_indices(qc, circuit_instruction.qubits)

    if name in ("cx", "cnot"):
        return [
            {
                "gate": "CNOT",
                "targets": [(qubits[0], qubits[1])],
                "params": {},
            }
        ]
    if name == "cz":
        c, t = qubits[0], qubits[1]
        return [
            {"gate": "H", "targets": [t], "params": {}},
            {"gate": "CNOT", "targets": [(c, t)], "params": {}},
            {"gate": "H", "targets": [t], "params": {}},
        ]
    if name == "swap":
        a, b = qubits[0], qubits[1]
        return [
            {"gate": "CNOT", "targets": [(a, b)], "params": {}},
            {"gate": "CNOT", "targets": [(b, a)], "params": {}},
            {"gate": "CNOT", "targets": [(a, b)], "params": {}},
        ]

    # Single-qubit native gates
    if len(qubits) == 1:
        gate = _normalize_gate_name(name)
        params: dict = {}
        if gate == "Rz":
            if not op.params:
                theta = 0.0
            else:
                theta = float(op.params[0])
            # Normalize angle to (-pi, pi]
            theta = math.remainder(theta, 2 * math.pi)
            params["theta"] = theta
        return [{"gate": gate, "targets": [qubits[0]], "params": params}]

    raise ValueError(f"Unsupported QASM gate: {op.name}")


def qiskit_circuit_to_ir(qc: QuantumCircuit) -> list[dict]:
    """Convert a Qiskit circuit to flat sequential instruction dicts."""
    instructions: list[dict] = []
    for circuit_instruction in qc.data:
        instructions.extend(_instruction_to_ir(qc, circuit_instruction))
    return instructions


def load_qasm_circuit(qasm: str) -> list[dict]:
    """Parse an OpenQASM string into the internal instruction IR."""
    qc = QuantumCircuit.from_qasm_str(qasm)
    return qiskit_circuit_to_ir(qc)


def load_qasm_file(path: str | Path) -> list[dict]:
    """Parse an OpenQASM file into the internal instruction IR."""
    qc = QuantumCircuit.from_qasm_file(str(path))
    return qiskit_circuit_to_ir(qc)
