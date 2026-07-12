"""General circuit compilation via T-cultivation (Clifford + T)."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from src.ds import get_microarchitecture
from src.tfim_layer_log import build_clifford_layer_log
from src.t_cultivation.t_cultivation import t_cultivation_execution
from src.t_cultivation.util import expand_multi_target_layers
from src.util import (
    analyze_execution_log,
    clifford_circuit_time_from_execution_log,
    execution_log_wall_time,
)

_T_NON_CLIFFORD = frozenset({"Rz", "T", "Tdg"})
_T_CLIFFORD = frozenset({"CNOT", "H", "S", "Sdg", "X", "Y", "Z", "CZ", "SWAP", "I"})


def build_t_exec_kwargs(
    *,
    n_qubits: int,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    code_distance: int,
    config: dict,
) -> dict[str, Any]:
    rng = config["rng"]
    if not isinstance(rng, np.random.Generator):
        raise TypeError("config['rng'] must be a numpy.random.Generator")

    exec_kwargs: dict[str, Any] = {
        "n_factories": n_qubits,
        "logic_qubit_locations": logic_qubit_locations,
        "magic_state_locations": magic_state_locations,
        "rng": rng,
        "n_aods": config["n_aods"],
        "to_decompose": config.get("to_decompose", False),
        "print_profile": config.get("print_profile", False),
        "epsilon": config.get("epsilon", 1e-4),
        "code_distance": code_distance,
        "trivial_return": bool(config.get("trivial_return", False)),
        "decompose_move": bool(config.get("decompose_move", False)),
        "redistribute_stage1_success": bool(
            config.get("redistribute_stage1_success", True)
        ),
    }
    if "num_subfactories" in config:
        exec_kwargs["num_subfactories"] = config["num_subfactories"]
    if "synchronize_factory_execution" in config:
        exec_kwargs["synchronize_factory_execution"] = config[
            "synchronize_factory_execution"
        ]
    if "logical_se_interval" in config:
        exec_kwargs["logical_se_interval"] = config["logical_se_interval"]
    return exec_kwargs


def _clifford_layer_time_before_rz(
    circuit: list[dict],
    full_logs: list[list[dict]],
    round_idx: int,
) -> float:
    rz_seen = 0
    rz_pos: int | None = None
    for i, instr in enumerate(circuit):
        if instr["gate"] not in _T_NON_CLIFFORD:
            continue
        if instr["gate"] != "Rz":
            continue
        if rz_seen == round_idx:
            rz_pos = i
            break
        rz_seen += 1
    if rz_pos is None:
        return 0.0

    clifford_time = 0.0
    j = rz_pos - 1
    while j >= 0 and circuit[j]["gate"] in _T_CLIFFORD:
        clifford_time += execution_log_wall_time(full_logs[j])
        j -= 1
    return clifford_time


def _build_t_profiling_row(
    *,
    n_qubits: int,
    qubit_layout: tuple,
    round_idx: int,
    code_distance: int,
    placement: str,
    n_aods: int,
    execution_log: list[dict],
    profiling_result: dict[str, Any],
    config: dict[str, Any],
    clifford_layer_time: float = 0.0,
) -> dict[str, Any]:
    ops = profiling_result["ops"]

    def _op_circuit_time(name: str) -> float:
        return float(ops.get(name, {}).get("circuit_time", 0.0))

    qubit_cnot_counts = profiling_result.get("qubit_cnot_counts") or []
    n_cnot = int(sum(qubit_cnot_counts))
    max_rus = max(qubit_cnot_counts) if qubit_cnot_counts else 0
    avg_rus = (
        sum(qubit_cnot_counts) / len(qubit_cnot_counts) if qubit_cnot_counts else 0.0
    )
    return {
        "n_qubits": n_qubits,
        "qubit_cols": qubit_layout[0],
        "qubit_rows": qubit_layout[1],
        "round": round_idx,
        "code_distance": code_distance,
        "placement": placement,
        "n_aods": n_aods,
        "trivial_return": bool(config.get("trivial_return", False)),
        "decompose_move": bool(config.get("decompose_move", False)),
        "redistribute_stage1_success": bool(
            config.get("redistribute_stage1_success", True)
        ),
        "total_time": profiling_result["total_time"],
        "movement_time": _op_circuit_time("move"),
        "return_movement_time": _op_circuit_time("return_move"),
        "stage1_time": _op_circuit_time("SE_stage_1"),
        "stage2_time": _op_circuit_time("SE_stage_2"),
        "clifford_time": clifford_layer_time
        + clifford_circuit_time_from_execution_log(execution_log),
        "TMR_round": _op_circuit_time("Rz"),
        "RUS_round": _op_circuit_time("CNOT"),
        "n_cnot": n_cnot,
        "max_rus_per_qubit": max_rus,
        "avg_rus_per_qubit": avg_rus,
        "initial_angle": profiling_result.get("initial_angle"),
        "largest_angle": profiling_result.get("largest_angle"),
        "tmr_total": profiling_result["failures"]["tmr_total"],
        "rus_total": profiling_result["failures"]["rus_total"],
    }


def _compile_split_layers(
    circuit: list[dict],
    *,
    logic_qubit_locations: list[tuple[int, int]],
    exec_kwargs: dict[str, Any],
) -> list[list[dict]]:
    full_logs: list[list[dict]] = []
    for instruction in circuit:
        gate = instruction["gate"]
        if gate == "Rz" or gate in ("T", "Tdg"):
            nc_log = t_cultivation_execution(
                circuit=[instruction],
                **exec_kwargs,
            )
            full_logs.append(nc_log)
            instruction["t_layer_log"] = nc_log
            instruction["t_layer_log_type"] = "non_clifford"
        elif gate in _T_CLIFFORD:
            clifford_log = build_clifford_layer_log(
                instruction=instruction,
                logic_qubit_locations=logic_qubit_locations,
            )
            full_logs.append(clifford_log)
            instruction["t_layer_log"] = clifford_log
            instruction["t_layer_log_type"] = "clifford"
        else:
            raise ValueError(f"T-cultivation compile does not support gate '{gate}'")
    return full_logs


def compile_circuit_t_cultivation(
    circuit: list[dict],
    *,
    n_qubits: int,
    qubit_layout: tuple,
    placement: str,
    code_distance: int,
    config: dict,
    split_layers: bool = True,
    analyze_result: bool = False,
    result_path: str | None = None,
    logic_qubit_locations: list[tuple[int, int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[list[dict]], list[dict[str, Any]]]:
    """Compile a Clifford+T (or Rz-decomposed) circuit with T-cultivation."""
    if logic_qubit_locations is None or magic_state_locations is None:
        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories=n_qubits,
            qubit_layout=qubit_layout,
            placement=placement,
        )

    exec_kwargs = build_t_exec_kwargs(
        n_qubits=n_qubits,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        code_distance=code_distance,
        config=config,
    )
    n_aods = config["n_aods"]

    if split_layers:
        full_logs = _compile_split_layers(
            circuit,
            logic_qubit_locations=logic_qubit_locations,
            exec_kwargs=exec_kwargs,
        )
    else:
        epsilon = exec_kwargs.get("epsilon", 1e-4)
        to_decompose = exec_kwargs.get("to_decompose", False)
        expanded = expand_multi_target_layers(circuit, epsilon, to_decompose)
        full_log = t_cultivation_execution(circuit=expanded, **exec_kwargs)
        full_logs = [full_log]
        for instruction in circuit:
            instruction["t_layer_log"] = full_log
            instruction["t_layer_log_type"] = "full_circuit"

    profiling_results: list[dict[str, Any]] = []
    if analyze_result and split_layers:
        rz_round = 0
        for instruction, log in zip(circuit, full_logs):
            if instruction["gate"] != "Rz":
                continue
            profiling_result = analyze_execution_log(log, n_factories=n_qubits)
            csv_result = _build_t_profiling_row(
                n_qubits=n_qubits,
                qubit_layout=qubit_layout,
                round_idx=rz_round,
                code_distance=code_distance,
                placement=placement,
                n_aods=n_aods,
                execution_log=log,
                profiling_result=profiling_result,
                config=config,
                clifford_layer_time=_clifford_layer_time_before_rz(
                    circuit, full_logs, rz_round
                ),
            )
            profiling_results.append(csv_result)
            rz_round += 1
    elif analyze_result and not split_layers and full_logs:
        profiling_result = analyze_execution_log(full_logs[0], n_factories=n_qubits)
        profiling_results.append(
            _build_t_profiling_row(
                n_qubits=n_qubits,
                qubit_layout=qubit_layout,
                round_idx=0,
                code_distance=code_distance,
                placement=placement,
                n_aods=n_aods,
                execution_log=full_logs[0],
                profiling_result=profiling_result,
                config=config,
            )
        )

    if result_path is not None:
        with open(result_path, "wb") as f:
            pickle.dump({"circuit": circuit, "full_logs": full_logs}, f)

    return circuit, full_logs, profiling_results
