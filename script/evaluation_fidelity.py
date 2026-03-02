from src.fidelity_simulation import (
    simluate_trotter_2d_tfim_fidelity,
    simluate_trotter_2d_tfim_fidelity_star,
)
from src.error_model import PhysicalErrorModel, LogicalErrorModel
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
import os
from itertools import product
from src.tfim_star import generate_one_layer_2d_tfim_circuit_star
import numpy as np
import csv


# ! only consider 1 trotter for now
def run_evaluation_raw(params: dict, physical_error_model: PhysicalErrorModel):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    print("=" * 80)
    print("EVALUATION SCRIPT - RAW FIDELITY SIMULATION")
    print("=" * 80 + "\n")
    # simulate raw physical fidelity for different qubit layouts and trotter steps
    for qubit_layout in params["qubit_layout"]:
        n_qubits = qubit_layout[0] * qubit_layout[1]
        for J, h, dt, n_trotter in params["tfim"]:
            qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
                n_qubits=n_qubits, qubit_layout=qubit_layout, J=J, h=h, dt=dt
            )
            # for n_trotter_steps in range(1, n_trotter):
            for n_trotter_steps in range(1, 2):
                fidelity_profile = simluate_trotter_2d_tfim_fidelity(
                    n_qubits=n_qubits,
                    qubit_layout=qubit_layout,
                    n_trotter_steps=n_trotter_steps,
                    qc_one_layer=qc_one_layer,
                    physical_error_model=physical_error_model,
                )
                result = {
                    "qubit_layout": qubit_layout,
                    "n_trotter_steps": n_trotter_steps,
                    **fidelity_profile,
                }
                results.append(result)

    # simulate raw physical fidelity for different qubit layouts and trotter steps

    # Save results to csv file
    results_path = os.path.join(output_dir, "raw_fidelity_results.csv")
    with open(results_path, "w") as f:
        # Write header
        f.write(
            "n_qubit,n_trotter_steps,fidelity,total_duration,fidelity_cz,fidelity_1q,fidelity_move,fidelity_idle,fidelity_init,fidelity_measurement\n"
        )
        for result in results:
            f.write(
                f"{result['qubit_layout'][0]*result['qubit_layout'][1]},{result['n_trotter_steps']},{result['fidelity']},{result['total_duration']},{result['fidelity_cz']},{result['fidelity_1q']},{result['fidelity_move']},{result['fidelity_idle']},{result['fidelity_init']},{result['fidelity_measurement']}\n"
            )


def run_evaluation_star(
    params: dict, logical_error_model: LogicalErrorModel, analyze_result: bool
):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)
    results = []
    profiling_results = []
    for (
        (n_cols, n_rows),
        (J, h, dt, n_trotter),
        placement,
        n_aods,
        skip_rus,
        trivial_ret,
        decompose_move,
        parallel_execution,
    ) in product(
        params["qubit_layout"],
        params["tfim"],
        params["placement_methods"],
        params["n_aods"],
        params["consider_skip_rus"],
        params["trivial_return"],
        params["decompose_move"],
        params["parallel_execution"],
    ):
        if parallel_execution and decompose_move:
            # For simplicity, we only evaluate parallel execution for non-decomposed movement
            continue
        for trial in range(params["trials_per_config"]):
            rng = np.random.default_rng(42 + trial)
            log_dir = f"output/evaluation/fidelty/results/{n_rows}x{n_cols}/{placement}/naod_{n_aods}/skipRUS_{skip_rus}/trivial_return_{trivial_ret}/decompos_move_{decompose_move}/parallel_{parallel_execution}"
            os.makedirs(log_dir, exist_ok=True)
            config = {
                "n_aods": n_aods,
                "consider_skip_rus": skip_rus,
                "trivial_return": trivial_ret,
                "decompose_move": decompose_move,
                "rng": rng,
                "save_log": False,
            }
            qc_one_layer, rz_logs, profiling_results_per_case = (
                generate_one_layer_2d_tfim_circuit_star(
                    n_qubits=n_cols * n_rows,
                    qubit_layout=(n_rows, n_cols),
                    placement=placement,
                    J=J,
                    h=h,
                    dt=dt,
                    code_distance=logical_error_model.code_distance,
                    config=config,
                    parallel_execution=parallel_execution,
                    analyze_result=analyze_result,
                    # result_path=log_dir + f"/trial_{trial}.pickle",
                )
            )
            profiling_results += profiling_results_per_case
            result = simluate_trotter_2d_tfim_fidelity_star(
                n_qubits=n_cols * n_rows,
                n_factories=n_cols * n_rows,
                qubit_layout=(n_rows, n_cols),
                n_trotter_steps=1,  # for simplicity, we only evaluate 1 trotter step for star compilation
                qc_one_layer=qc_one_layer,
                execution_logs=rz_logs,
                logical_error_model=logical_error_model,
            )
            results.append(
                {
                    "qubit_layout": (n_rows, n_cols),
                    "n_trotter_steps": 1,
                    "placement": placement,
                    "n_aods": n_aods,
                    "skip_rus": skip_rus,
                    "trivial_return": trivial_ret,
                    "decompose_move": decompose_move,
                    "parallel_execution": parallel_execution,
                    **result,
                }
            )

    # Save results to csv file
    results_path = os.path.join(output_dir, "star_fidelity_results.csv")
    with open(results_path, "w") as f:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    if analyze_result:
        profiling_results_path = os.path.join(
            output_dir, "star_full_trotter_profiling_results.csv"
        )
        with open(profiling_results_path, "a", newline="") as csvfile:
            fieldnames = profiling_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profiling_results)


if __name__ == "__main__":
    # Define parameter grid
    qubit_layout = [
        (4, 4),  # 16 qubits
        (5, 5),  # 25 qubits
        (6, 6),  # 36 qubits
        (7, 7),  # 49 qubits
        (8, 8),  # 64 qubits
        (9, 9),  # 81 qubits
        (10, 10),  # 100 qubits
    ]
    physical_error_model: PhysicalErrorModel = PhysicalErrorModel("lookahead")
    p_ph = physical_error_model.get_error_rate("p_ph")
    j_h = [(1.0, 1.0)]  # J, h, dt
    tfim = []
    l1 = 1
    alpha = 2
    omega = 1
    for j, h in j_h:
        dt = l1 * alpha * p_ph / omega
        T = 10 / j
        n_trotter = int(T / dt)
        tfim.append((j, h, dt, n_trotter))

    params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
    }
    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": [
            "col_based",
            "checkerboard",
        ],
        "n_aods": [1, 2, 3, 4, 5],
        "consider_skip_rus": [0, 1, 2],
        "trivial_return": [False, True],
        "trials_per_config": 10,
        "decompose_move": [False, True],
        "parallel_execution": [False, True],
    }
    # run_evaluation_raw(params=params, physical_error_model=physical_error_model)

    logical_error_model = LogicalErrorModel(
        physical_model=physical_error_model, code_distance=7
    )

    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": [
            "col_based",
            "checkerboard",
        ],
        "n_aods": [1],
        "consider_skip_rus": [0],
        "trivial_return": [False],
        "trials_per_config": 1,
        "decompose_move": [False],
        "parallel_execution": [False],
    }

    run_evaluation_star(
        params=star_params, logical_error_model=logical_error_model, analyze_result=True
    )
