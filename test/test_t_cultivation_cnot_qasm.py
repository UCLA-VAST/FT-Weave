import numpy as np

from src.t_cultivation.config import update_config
from src.t_cultivation.general_circuit_t import compile_circuit_t_cultivation


def _build_cnot_t_circuit() -> list[dict]:
    return [
        {"gate": "H", "targets": [0], "params": {}},
        {"gate": "CNOT", "targets": [(0, 1)], "params": {}},
        {"gate": "T", "targets": [1], "params": {}},
        {"gate": "CNOT", "targets": [(1, 2)], "params": {}},
    ]


def test_t_cultivation_no_split_schedules_circuit_cnots():
    update_config(
        STAGE_2_FIDELITY_TARGET=1e-8,
        FACTORY_PHYSICAL_SIZE=2,
        STAGE_1_RESOURCE_UNITS=1,
    )
    circuit = _build_cnot_t_circuit()
    config = {
        "n_aods": 2,
        "rng": np.random.default_rng(42),
        "epsilon": 1e-4,
        "to_decompose": False,
        "trivial_return": False,
        "decompose_move": False,
        "redistribute_stage1_success": True,
    }
    _, full_logs, _ = compile_circuit_t_cultivation(
        circuit,
        n_qubits=3,
        qubit_layout=(2, 2),
        placement="col_based",
        code_distance=7,
        config=config,
        split_layers=False,
        analyze_result=False,
    )
    assert len(full_logs) == 1
    log = full_logs[0]

    circuit_cnots = [
        entry
        for entry in log
        if entry.get("operation") == "CNOT"
        and not entry.get("factories")
        and entry.get("move_vecs") is None
    ]
    assert len(circuit_cnots) >= 2
