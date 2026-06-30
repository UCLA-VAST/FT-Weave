from src.star.general_circuit_star import compile_circuit_star
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz


def generate_one_layer_2d_tfim_circuit_star(
    n_qubits: int,
    qubit_layout: tuple,
    placement: str,
    J: float,
    h: float,
    dt: float,
    code_distance: int,
    config: dict,
    parallel_execution: bool,
    analyze_result: bool,
    result_path: str | None = None,
) -> tuple[list[dict], list[list[tuple] | list[dict]], list[dict]]:
    qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        J=J,
        h=h,
        dt=dt,
        logical=True,
        order=2,
    )
    return compile_circuit_star(
        qc_one_layer,
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        code_distance=code_distance,
        config=config,
        parallel_execution=parallel_execution,
        analyze_result=analyze_result,
        result_path=result_path,
    )
