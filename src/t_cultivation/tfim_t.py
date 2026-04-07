"""TFIM one-layer circuit generation for T-cultivation.

Mirrors :mod:`src.star.tfim_star`: each **round** is one ``Rz`` instruction in the
logical layer, with a separate ``t_cultivation_execution`` run and log. ``CNOT`` /
``H`` instructions are not executed here; they enter fidelity only via analytic
Clifford counts from ``qc_one_layer`` (same as STAR).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from src.ds import get_microarchitecture
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
from src.t_cultivation.t_cultivation import t_cultivation_execution
from src.util import analyze_execution_log


def generate_one_layer_2d_tfim_circuit_t_cultivation(
    n_qubits: int,
    qubit_layout: tuple,
    placement: str,
    J: float,
    h: float,
    dt: float,
    code_distance: int,
    config: dict,
    analyze_result: bool,
    result_path: str | None = None,
    logic_qubit_locations: list[tuple[int, int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], list[list[tuple]], list[dict[str, Any]]]:
    """
    Build one logical TFIM Trotter layer and run T-cultivation **per Rz round**.

    Same control-flow as :func:`src.star.tfim_star.generate_one_layer_2d_tfim_circuit_star`:
    iterate ``qc_one_layer`` in order; for each ``Rz`` instruction, run scheduling once
    on that instruction only and append a log. ``CNOT`` / ``H`` entries are skipped
    (asserted) — execution logs come only from Rz rounds.

    ``config`` must include ``n_aods`` and ``rng`` (:class:`numpy.random.Generator`).
    The same ``rng`` is reused across rounds (it advances naturally each call).

    Returns:
        ``qc_one_layer``: full logical layer (same as STAR).
        ``rz_logs``: one execution log per Rz instruction, in circuit order.
        ``profiling_results``: when ``analyze_result``, one profiling row per round
        with ``round`` = 0, 1, ….
    """
    qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        J=J,
        h=h,
        dt=dt,
        logical=True,
        order=2,
    )

    if logic_qubit_locations is None or magic_state_locations is None:
        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories=n_qubits,
            qubit_layout=qubit_layout,
            placement=placement,
        )

    rng = config["rng"]
    if not isinstance(rng, np.random.Generator):
        raise TypeError("config['rng'] must be a numpy.random.Generator")
    n_aods = config["n_aods"]

    exec_kwargs: dict[str, Any] = {
        "n_factories": n_qubits,
        "logic_qubit_locations": logic_qubit_locations,
        "magic_state_locations": magic_state_locations,
        "rng": rng,
        "n_aods": n_aods,
        "to_decompose": config.get("to_decompose", False),
        "print_profile": config.get("print_profile", False),
        "epsilon": config.get("epsilon", 1e-4),
    }
    if "num_subfactories" in config:
        exec_kwargs["num_subfactories"] = config["num_subfactories"]
    if "synchronize_factory_execution" in config:
        exec_kwargs["synchronize_factory_execution"] = config[
            "synchronize_factory_execution"
        ]

    rz_logs: list[list[tuple]] = []
    for instruction in qc_one_layer:
        if instruction["gate"] == "Rz":
            rz_logs.append(
                t_cultivation_execution(
                    circuit=[instruction],
                    **exec_kwargs,
                )
            )
        else:
            assert instruction["gate"] in ("CNOT", "H")

    profiling_results: list[dict[str, Any]] = []
    if analyze_result:
        for i, log in enumerate(rz_logs):
            profiling_result = analyze_execution_log(log, n_factories=n_qubits)
            ops = profiling_result["ops"]

            def _op_circuit_time(name: str) -> float:
                return float(ops.get(name, {}).get("circuit_time", 0.0))

            qubit_cnot_counts = profiling_result.get("qubit_cnot_counts") or []
            max_rus = max(qubit_cnot_counts) if qubit_cnot_counts else 0
            avg_rus = (
                sum(qubit_cnot_counts) / len(qubit_cnot_counts)
                if qubit_cnot_counts
                else 0.0
            )

            csv_result = {
                "n_qubits": n_qubits,
                "qubit_cols": qubit_layout[0],
                "qubit_rows": qubit_layout[1],
                "round": i,
                "code_distance": code_distance,
                "placement": placement,
                "n_aods": n_aods,
                "total_time": profiling_result["total_time"],
                "movement_time": _op_circuit_time("move"),
                "return_movement_time": _op_circuit_time("return_move"),
                "TMR_round": _op_circuit_time("Rz"),
                "RUS_round": _op_circuit_time("CNOT"),
                "max_rus_per_qubit": max_rus,
                "avg_rus_per_qubit": avg_rus,
                "initial_angle": profiling_result.get("initial_angle"),
                "largest_angle": profiling_result.get("largest_angle"),
                "tmr_total": profiling_result["failures"]["tmr_total"],
                "rus_total": profiling_result["failures"]["rus_total"],
            }
            profiling_results.append(csv_result)

    if result_path is not None:
        with open(result_path, "wb") as f:
            pickle.dump(
                {
                    "qc_one_layer": qc_one_layer,
                    "rz_logs": rz_logs,
                },
                f,
            )

    return qc_one_layer, rz_logs, profiling_results
