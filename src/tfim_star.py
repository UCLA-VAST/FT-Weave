from src.analog_rotation_execution import factory_angle_execution
from src.analog_rotation_execution_parallel import factory_angle_execution_parallel
from src.ds import FactoryPool, get_microarchitecture
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
from src.util import analyze_execution_log
import pickle

import numpy as np
from scipy.optimize import brentq


# -------------------------------------------------------
# Forward mapping: physical -> logical
# -------------------------------------------------------
def logical_from_physical(physical_angle: float, d: int) -> float:
    """
    Compute logical angle θ from physical angle θ*.
    """
    s = np.sin(physical_angle / 2)
    c = np.cos(physical_angle / 2)

    numerator = s**d
    denominator = np.sqrt(s ** (2 * d) + c ** (2 * d))

    return 2 * np.arcsin(numerator / denominator)


# -------------------------------------------------------
# Inverse mapping: logical -> physical
# -------------------------------------------------------
def convert_logical_angle_to_physical_angle(
    logical_angle: float, d: int, tol: float = 1e-12
) -> float:
    """
    Numerically solve for physical angle θ*
    given logical angle θ.
    """

    # Root function:
    def root_fn(physical_angle):
        return logical_from_physical(physical_angle, d) - logical_angle

    # Physical angle lies in [0, π]
    physical_angle, result = brentq(root_fn, 0.0, np.pi, xtol=tol, full_output=True)
    recovered_logical_angle = logical_from_physical(physical_angle, d)
    assert np.isclose(
        logical_angle, recovered_logical_angle, atol=tol
    ), f"Failed to recover logical angle: expected {logical_angle}, got {recovered_logical_angle}"
    return physical_angle


def generate_one_layer_2d_tfim_circuit_star(
    n_qubits: int,
    qubit_layout: tuple,
    J: float,
    h: float,
    dt: float,
    code_distance: int,
    config: dict,
    parallel_execution: bool,
    analyze_result: bool,
    result_path: str | None = None,
) -> tuple[list[dict], list[list[tuple]], list[dict]]:
    qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
        n_qubits=n_qubits, qubit_layout=qubit_layout, J=J, h=h, dt=dt, logical=True
    )
    # add logic_qubit_locations, and magic_state_locations here
    rz_logs: list[list[tuple]] = []
    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits,
        n_factories=n_qubits,
        qubit_layout=qubit_layout,
        placement=config["placement"],
    )
    for instruction in qc_one_layer:
        # contruct target_qubits_angles
        physical_angle = convert_logical_angle_to_physical_angle(
            logical_angle=instruction["params"]["theta"],
            d=code_distance,
        )
        target_qubits_angles = {
            target: physical_angle for target in instruction["targets"]
        }
        if instruction["gate"] == "Rz":
            # construct factory_pool
            factory_pool = FactoryPool(num_factories=n_qubits)
            if parallel_execution:
                func = factory_angle_execution_parallel
            else:
                func = factory_angle_execution
            time, rz_log = func(
                target_qubits_angles=target_qubits_angles,
                logic_qubit_locations=logic_qubit_locations,
                magic_state_locations=magic_state_locations,
                factory_pool=factory_pool,
                code_distance=code_distance,
                **config,
            )
            rz_logs.append(rz_log)
            # collect target_qubits_angles
            raise NotImplementedError(
                "Rz gates should be executed using factory_angle_execution or factory_angle_execution_parallel"
            )
        else:
            assert instruction["gate"] in ["CNOT", "H"]

    profiling_results = []
    if analyze_result:
        for i, log in enumerate(rz_logs):
            profiling_result = analyze_execution_log(log, n_factories=n_qubits)
            csv_result = {
                "n_qubits": n_qubits,
                "qubit_cols": qubit_layout[0],
                "qubit_rows": qubit_layout[1],
                "round": i,
                "placement": config["placement"],
                "n_aods": config["n_aods"],
                "consider_skip_rus": config["consider_skip_rus"],
                "tmr_assignment_method": "matching",
                "trivial_return": config["trivial_return"],
                "decompose_move": config["decompose_move"],
                "parallel_execution": parallel_execution,
                "total_time": profiling_result["total_time"],
                "movement_time": profiling_result["ops"]["move"]["circuit_time"],
                "return_movement_time": profiling_result["ops"]["return_move"][
                    "circuit_time"
                ],
                "TMR_round": profiling_result["ops"]["Rz"]["circuit_time"],
                "RUS_round": profiling_result["ops"]["CNOT"]["circuit_time"],
                "max_rus_per_qubit": max(profiling_result["qubit_cnot_counts"]),
                "avg_rus_per_qubit": sum(profiling_result["qubit_cnot_counts"])
                / len(profiling_result["qubit_cnot_counts"]),
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
