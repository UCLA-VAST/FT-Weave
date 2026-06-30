"""TFIM one-layer circuit generation for T-cultivation.

Mirrors :mod:`src.star.tfim_star`: each **round** is one ``Rz`` instruction in the
logical layer, with a separate ``t_cultivation_execution`` run and log. ``CNOT`` /
``H`` instructions are not executed here; they enter fidelity only via analytic
Clifford counts from ``qc_one_layer`` (same as STAR).
"""

from __future__ import annotations

from typing import Any

from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
from src.t_cultivation.general_circuit_t import compile_circuit_t_cultivation


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
) -> tuple[list[dict], list[list[dict]], list[dict[str, Any]]]:
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
        ``full_logs``: one execution log per logical instruction, in circuit order.
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
    return compile_circuit_t_cultivation(
        qc_one_layer,
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        code_distance=code_distance,
        config=config,
        split_layers=True,
        analyze_result=analyze_result,
        result_path=result_path,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
    )
