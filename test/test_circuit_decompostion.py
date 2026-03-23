from src.t_cultivation.util import expand_multi_target_layers


def test_expand_multi_target_layers_keeps_non_rz_when_not_decomposing():
    circuit = [
        {"gate": "H", "targets": [0, 1], "params": {}},
        {"gate": "CNOT", "targets": [(0, 2), (1, 3)], "params": {}},
    ]

    expanded = expand_multi_target_layers(circuit, to_decompose=False)

    assert len(expanded) == 2
    assert expanded[0] == circuit[0]
    assert expanded[1] == circuit[1]


def test_expand_multi_target_layers_always_decomposes_rz(monkeypatch):
    def fake_gridsynth_rz_templates(theta: float):
        assert theta == 0.5
        return [
            {"gate": "H", "params": {}},
            {"gate": "T", "params": {}},
        ]

    monkeypatch.setattr(
        "src.t_cultivation.util.gridsynth_rz_templates", fake_gridsynth_rz_templates
    )

    circuit = [
        {"gate": "H", "targets": [9], "params": {}},
        {
            "gate": "Rz",
            "targets": [0, 1],
            "params": {"theta": 0.5},
        },
    ]

    expanded = expand_multi_target_layers(circuit, to_decompose=False)
    assert len(expanded) == 5
    assert expanded[0] == circuit[0]

    assert expanded[1]["gate"] == "H"
    assert expanded[1]["targets"] == [0]

    assert expanded[2]["gate"] == "T"
    assert expanded[2]["targets"] == [0]

    assert expanded[3]["gate"] == "H"
    assert expanded[3]["targets"] == [1]

    assert expanded[4]["gate"] == "T"


def test_expand_multi_target_layers_expands_non_rz_when_decomposing():
    circuit = [
        {"gate": "H", "targets": [0, 1], "params": {}},
        {"gate": "CNOT", "targets": [(0, 2), (1, 3)], "params": {}},
    ]

    expanded = expand_multi_target_layers(circuit, to_decompose=True)

    assert len(expanded) == 4

    assert expanded[0]["gate"] == "H"
    assert expanded[0]["targets"] == [0]

    assert expanded[1]["gate"] == "H"
    assert expanded[1]["targets"] == [1]

    assert expanded[2]["gate"] == "CNOT"
    assert expanded[2]["targets"] == [0, 2]

    assert expanded[3]["gate"] == "CNOT"
    assert expanded[3]["targets"] == [1, 3]
