"""Logical fidelity estimation for T-cultivation execution of TFIM layers."""

from __future__ import annotations

from src.error_model import LogicalErrorModel
from src.fidelity_simulation.circuit_fidelity import (
    simulate_t_cultivation_circuit_fidelity,
)


def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    qc_one_layer: list[dict],
    execution_logs: list[list[dict]],
    logical_error_model: LogicalErrorModel,
    synthesis_epsilon: float | None = None,
) -> dict:
    """TFIM T-cultivation fidelity wrapper (delegates to general circuit simulator)."""
    _ = n_qubits
    _ = qubit_layout
    return simulate_t_cultivation_circuit_fidelity(
        n_factories=n_factories,
        circuit=qc_one_layer,
        execution_logs=execution_logs,
        logical_error_model=logical_error_model,
        synthesis_epsilon=synthesis_epsilon,
        n_trotter_steps=n_trotter_steps,
    )
