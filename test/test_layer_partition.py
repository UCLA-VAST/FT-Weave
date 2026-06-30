import math

from src.circuit.layer_partition import partition_into_layers


def test_partition_merges_same_angle_rz():
    flat = [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.5}},
        {"gate": "Rz", "targets": [1], "params": {"theta": 0.5}},
        {"gate": "CNOT", "targets": [(0, 1)], "params": {}},
    ]
    layers = partition_into_layers(flat)
    assert len(layers) == 3
    assert layers[0]["gate"] == "H"
    assert layers[1]["gate"] == "Rz"
    assert layers[1]["targets"] == [0, 1]
    assert layers[1]["params"]["theta"] == 0.5
    assert layers[2]["gate"] == "CNOT"


def test_partition_merges_different_angle_rz():
    flat = [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.5}},
        {"gate": "Rz", "targets": [1], "params": {"theta": 0.3}},
        {"gate": "CNOT", "targets": [(0, 1)], "params": {}},
    ]
    layers = partition_into_layers(flat)
    assert len(layers) == 3
    assert layers[1]["gate"] == "Rz"
    assert set(layers[1]["targets"]) == {0, 1}
    assert layers[1]["params"]["angles"] == {0: 0.5, 1: 0.3}
    assert "theta" not in layers[1]["params"]


def test_partition_splits_rz_across_clifford_blocks():
    flat = [
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.5}},
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "Rz", "targets": [1], "params": {"theta": 0.3}},
    ]
    layers = partition_into_layers(flat)
    assert len(layers) == 3
    assert layers[0]["gate"] == "Rz"
    assert layers[1]["gate"] == "H"
    assert layers[2]["gate"] == "Rz"


def test_partition_sums_repeated_rz_on_same_qubit():
    flat = [
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.5}},
        {"gate": "Rz", "targets": [0], "params": {"theta": 0.3}},
    ]
    layers = partition_into_layers(flat)
    assert len(layers) == 1
    assert math.isclose(layers[0]["params"]["theta"], 0.8)


def test_partition_preserves_clifford_order():
    flat = [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "H", "targets": [1], "params": {}},
    ]
    layers = partition_into_layers(flat)
    assert len(layers) == 2
    assert layers[0]["targets"] == [0]
    assert layers[1]["targets"] == [1]
