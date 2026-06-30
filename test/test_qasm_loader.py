import math
import tempfile
from pathlib import Path

from src.circuit.gate_set import classify_circuit, GateSetKind
from src.circuit.qasm_loader import load_qasm_circuit, load_qasm_file


SAMPLE_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
rz(0.5) q[1];
"""


def test_load_qasm_string():
    circuit = load_qasm_circuit(SAMPLE_QASM)
    assert len(circuit) >= 3
    assert circuit[0]["gate"] == "H"
    assert circuit[1]["gate"] == "CNOT"
    rz_gates = [g for g in circuit if g["gate"] == "Rz"]
    assert len(rz_gates) == 1
    assert math.isclose(rz_gates[0]["params"]["theta"], 0.5)
    assert classify_circuit(circuit) == GateSetKind.CLIFFORD_RZ


def test_load_qasm_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.qasm"
        path.write_text(SAMPLE_QASM)
        circuit = load_qasm_file(path)
        assert any(instr["gate"] == "Rz" for instr in circuit)
