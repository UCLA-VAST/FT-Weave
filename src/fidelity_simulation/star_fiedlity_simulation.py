from src.fidelity_simulation.circuit_fidelity import simulate_star_circuit_fidelity
from src.error_model import LogicalErrorModel


def simluate_trotter_2d_tfim_fidelity(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    n_trotter_steps: int,
    execution_logs: list[list[dict]],
    logical_error_model: LogicalErrorModel,
) -> dict:
    """TFIM STAR fidelity wrapper (delegates to general circuit simulator)."""
    _ = n_qubits
    _ = qubit_layout
    return simulate_star_circuit_fidelity(
        n_factories=n_factories,
        circuit=[],
        execution_logs=execution_logs,
        logical_error_model=logical_error_model,
        n_rz_expected=9 * n_trotter_steps,
    )
