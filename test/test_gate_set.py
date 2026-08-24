from src.circuit.gate_set import (
    GateSetKind,
    classify_circuit,
    validate_circuit_for_backend,
)


def test_classify_clifford_rz():
    circuit = [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.5}},
    ]
    assert classify_circuit(circuit) == GateSetKind.CLIFFORD_RZ


def test_classify_clifford_t():
    circuit = [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
    ]
    assert classify_circuit(circuit) == GateSetKind.CLIFFORD_T


def test_classify_pure_clifford():
    circuit = [{"gate": "H", "targets": [0], "params": {}}]
    assert classify_circuit(circuit) == GateSetKind.PURE_CLIFFORD


def test_validate_star_rejects_t_gates():
    circuit = [{"gate": "T", "targets": [0], "params": {}}]
    try:
        validate_circuit_for_backend(circuit, backend="star")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "STAR" in str(exc)


def test_validate_auto_selects_rz_for_star():
    circuit = [
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.1}},
    ]
    kind = validate_circuit_for_backend(circuit, backend="auto")
    assert kind == GateSetKind.CLIFFORD_RZ
