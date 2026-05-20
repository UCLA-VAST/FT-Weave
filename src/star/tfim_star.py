from src.star.analog_rotation_execution import factory_angle_execution
from src.star.analog_rotation_execution_parallel import factory_angle_execution_parallel
from src.ds import FactoryPool, get_microarchitecture
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
from src.tfim_layer_log import build_clifford_layer_log
from src.util import analyze_execution_log
import pickle


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
    target_qubits_angles = {
        target: instruction["params"]["theta"] for target in instruction["targets"]
    }
    factory_pool = FactoryPool(num_factories=n_qubits)
    func = factory_angle_execution_parallel if parallel_execution else factory_angle_execution
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
        "max_rus_per_qubit": max(profiling_result["qubit_cnot_counts"]),
        "avg_rus_per_qubit": sum(profiling_result["qubit_cnot_counts"])
        / len(profiling_result["qubit_cnot_counts"]),
        "initial_angle": profiling_result["initial_angle"],
        "largest_angle": profiling_result["largest_angle"],
        "tmr_total": profiling_result["failures"]["tmr_total"],
    }


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
    # for instruction in qc_one_layer:
    #     if instruction["gate"] == "Rz":
    #         print(
    #             f"Rz gate on qubits {instruction['targets']} with angle {instruction['params']['theta']:.4f} radians"
    #         )
    # input()
    # add logic_qubit_locations, and magic_state_locations here
    layer_logs: list[list[dict] | list[tuple]] = []
    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits,
        n_factories=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
    )
    for instruction in qc_one_layer:
        if instruction["gate"] == "Rz":
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
        else:
            assert instruction["gate"] in ["CNOT", "H"]
            clifford_log = build_clifford_layer_log(
                instruction=instruction,
                logic_qubit_locations=logic_qubit_locations,
            )
            layer_logs.append(clifford_log)
            instruction["star_layer_log"] = clifford_log
            instruction["star_layer_log_type"] = "clifford"

    profiling_results = []
    if analyze_result:
        rz_round = 0
        for instruction, log in zip(qc_one_layer, layer_logs):
            if instruction["gate"] != "Rz":
                continue
            profiling_result = analyze_execution_log(log, n_factories=n_qubits)
            # if "return_move" not in profiling_result["ops"]:
            #     for l in log:
            #         print(l)
            #     for key, value in profiling_result["ops"].items():
            #         print(f"{key}: {value}")
            #     raise ValueError("Missing return movement time in profiling result")
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

            pickle.dump(
                {
                    "qc_one_layer": qc_one_layer,
                    "layer_logs": layer_logs,
                },
                f,
            )
    return qc_one_layer, layer_logs, profiling_results
