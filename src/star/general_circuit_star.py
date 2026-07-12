"""General circuit compilation via STAR (Clifford + Rz)."""

from __future__ import annotations

import pickle
from typing import Any

from src.circuit.rz_params import rz_target_angles
from src.ds import FactoryPool, get_microarchitecture
from src.star.analog_rotation_execution import factory_angle_execution
from src.star.analog_rotation_execution_parallel import factory_angle_execution_parallel
from src.tfim_layer_log import build_clifford_layer_log
from src.util import analyze_execution_log


def _build_rz_round_log(
    *,
    instruction: dict,
    n_qubits: int,
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    code_distance: int,
    parallel_execution: bool,
    config: dict,
) -> list[dict]:
    target_qubits_angles = rz_target_angles(instruction)
    factory_pool = FactoryPool(num_factories=n_qubits)
    func = (
        factory_angle_execution_parallel
        if parallel_execution
        else factory_angle_execution
    )
    _, rz_log = func(
        target_qubits_angles=target_qubits_angles,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        factory_pool=factory_pool,
        code_distance=code_distance,
        **config,
    )
    return rz_log


def _build_star_profiling_row(
    *,
    n_qubits: int,
    qubit_layout: tuple,
    round_idx: int,
    code_distance: int,
    placement: str,
    config: dict,
    parallel_execution: bool,
    profiling_result: dict,
) -> dict:
    return {
        "n_qubits": n_qubits,
        "qubit_cols": qubit_layout[0],
        "qubit_rows": qubit_layout[1],
        "round": round_idx,
        "code_distance": code_distance,
        "placement": placement,
        "n_aods": config["n_aods"],
        "consider_skip_rus": config["consider_skip_rus"],
        "tmr_assignment_method": "matching",
        "trivial_return": config["trivial_return"],
        "decompose_move": config["decompose_move"],
        "prepare_lookahead_angles": config.get("prepare_lookahead_angles", True),
        "parallel_execution": parallel_execution,
        "total_time": profiling_result["total_time"],
        "movement_time": profiling_result["ops"]["move"]["circuit_time"],
        "return_movement_time": profiling_result["ops"]["return_move"]["circuit_time"],
        "TMR_round": profiling_result["ops"]["Rz"]["circuit_time"],
        "RUS_round": profiling_result["ops"]["CNOT"]["circuit_time"],
        "n_cnot": int(sum(profiling_result["qubit_cnot_counts"])),
        "max_rus_per_qubit": max(profiling_result["qubit_cnot_counts"]),
        "avg_rus_per_qubit": sum(profiling_result["qubit_cnot_counts"])
        / len(profiling_result["qubit_cnot_counts"]),
        "initial_angle": profiling_result["initial_angle"],
        "largest_angle": profiling_result["largest_angle"],
        "tmr_total": profiling_result["failures"]["tmr_total"],
    }


_STAR_CLIFFORD_GATES = frozenset({"CNOT", "H", "S", "Sdg", "X", "Y", "Z", "CZ", "SWAP", "I"})


def compile_circuit_star(
    circuit: list[dict],
    *,
    n_qubits: int,
    qubit_layout: tuple,
    placement: str,
    code_distance: int,
    config: dict,
    parallel_execution: bool,
    analyze_result: bool,
    result_path: str | None = None,
    logic_qubit_locations: list[tuple[int, int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[list[dict]], list[dict]]:
    """Compile a partitioned Clifford+Rz circuit with STAR.

    Returns:
        ``circuit`` (annotated with layer logs), ``layer_logs``, ``profiling_results``.
    """
    if logic_qubit_locations is None or magic_state_locations is None:
        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories=n_qubits,
            qubit_layout=qubit_layout,
            placement=placement,
        )

    layer_logs: list[list[dict]] = []
    for instruction in circuit:
        gate = instruction["gate"]
        if gate == "Rz":
            rz_log = _build_rz_round_log(
                instruction=instruction,
                n_qubits=n_qubits,
                logic_qubit_locations=logic_qubit_locations,
                magic_state_locations=magic_state_locations,
                code_distance=code_distance,
                parallel_execution=parallel_execution,
                config=config,
            )
            layer_logs.append(rz_log)
            instruction["star_layer_log"] = rz_log
            instruction["star_layer_log_type"] = "rz"
        elif gate in _STAR_CLIFFORD_GATES:
            clifford_log = build_clifford_layer_log(
                instruction=instruction,
                logic_qubit_locations=logic_qubit_locations,
            )
            layer_logs.append(clifford_log)
            instruction["star_layer_log"] = clifford_log
            instruction["star_layer_log_type"] = "clifford"
        else:
            raise ValueError(
                f"STAR compile does not support gate '{gate}' in circuit layer"
            )

    profiling_results: list[dict] = []
    if analyze_result:
        rz_round = 0
        for instruction, log in zip(circuit, layer_logs):
            if instruction["gate"] != "Rz":
                continue
            profiling_result = analyze_execution_log(log, n_factories=n_qubits)
            csv_result = _build_star_profiling_row(
                n_qubits=n_qubits,
                qubit_layout=qubit_layout,
                round_idx=rz_round,
                code_distance=code_distance,
                placement=placement,
                config=config,
                parallel_execution=parallel_execution,
                profiling_result=profiling_result,
            )
            profiling_results.append(csv_result)
            rz_round += 1

    if result_path is not None:
        with open(result_path, "wb") as f:
            pickle.dump({"circuit": circuit, "layer_logs": layer_logs}, f)

    return circuit, layer_logs, profiling_results
