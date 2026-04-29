from src.fidelity_simulation import (
    simluate_trotter_2d_tfim_fidelity,
    simluate_trotter_2d_tfim_fidelity_star,
)
from src.error_model import PhysicalErrorModel, LogicalErrorModel
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
import os
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
import numpy as np
import csv


SETTINGS = [
    (True, "matching", 0, False, False),
    (False, "matching", 0, False, False),
    (False, "matching", 1, False, False),
    (False, "matching", 2, False, False),
    (False, "matching", 2, True, False),
    (False, "matching", 2, False, True),
]


# ! only consider 1 trotter for now
def run_evaluation_raw(params: dict, physical_error_model: PhysicalErrorModel):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("EVALUATION SCRIPT - RAW FIDELITY SIMULATION")
    print("=" * 80 + "\n")
    print("Current run settings:")
    print(f"  qubit_layout: {params.get('qubit_layout')}")
    print(f"  tfim: {params.get('tfim')}")
    print(f"  physical_error_model: {physical_error_model}")

    results_path = os.path.join(output_dir, "raw_fidelity_results.csv")
    raw_header_needed = (not os.path.exists(results_path)) or (
        os.path.getsize(results_path) == 0
    )
    with open(results_path, "a", newline="") as csvfile:
        fieldnames = [
            "n_qubit",
            "n_trotter_steps",
            "fidelity",
            "total_duration",
            "fidelity_cz",
            "fidelity_1q",
            "fidelity_move",
            "fidelity_idle",
            "fidelity_init",
            "fidelity_measurement",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if raw_header_needed:
            writer.writeheader()

        # simulate raw physical fidelity for different qubit layouts and trotter steps
        for qubit_layout in params["qubit_layout"]:
            n_qubits = qubit_layout[0] * qubit_layout[1]
            for J, h, dt, n_trotter in params["tfim"]:
                qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
                    n_qubits=n_qubits,
                    qubit_layout=qubit_layout,
                    J=J,
                    h=h,
                    dt=dt,
                    order=2,
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
                    writer.writerow(
                        {
                            "n_qubit": n_qubits,
                            "n_trotter_steps": n_trotter_steps,
                            "fidelity": fidelity_profile["fidelity"],
                            "total_duration": fidelity_profile["total_duration"],
                            "fidelity_cz": fidelity_profile["fidelity_cz"],
                            "fidelity_1q": fidelity_profile["fidelity_1q"],
                            "fidelity_move": fidelity_profile["fidelity_move"],
                            "fidelity_idle": fidelity_profile["fidelity_idle"],
                            "fidelity_init": fidelity_profile["fidelity_init"],
                            "fidelity_measurement": fidelity_profile[
                                "fidelity_measurement"
                            ],
                        }
                    )


def run_evaluation_star(params: dict, logical_error_models, analyze_result: bool):
    """Run comprehensive evaluation with different parameter combinations."""

    # Support evaluating one or multiple code distances in a single run.
    if not isinstance(logical_error_models, (list, tuple)):
        logical_error_models = [logical_error_models]

    print("=" * 80)
    print("EVALUATION SCRIPT - STAR FIDELITY SIMULATION")
    print("=" * 80)
    print("Current run settings:")
    print(f"  qubit_layout: {params.get('qubit_layout')}")
    print(f"  tfim: {params.get('tfim')}")
    print(f"  placement_methods: {params.get('placement_methods')}")
    print(f"  n_aods: {params.get('n_aods')}")
    print(f"  settings_count: {len(params.get('settings', []))}")
    print(f"  trials_per_config: {params.get('trials_per_config')}")
    print(
        "  code_distances:",
        [model.code_distance for model in logical_error_models],
    )
    print()

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "star_fidelity_results.csv")
    profiling_results_path = os.path.join(
        output_dir, "star_full_trotter_profiling_results.csv"
    )

    result_writer = None
    profiling_writer = None
    result_header_needed = (not os.path.exists(results_path)) or (
        os.path.getsize(results_path) == 0
    )
    profiling_header_needed = (not os.path.exists(profiling_results_path)) or (
        os.path.getsize(profiling_results_path) == 0
    )

    result_file = open(results_path, "a", newline="")
    profiling_file = (
        open(profiling_results_path, "a", newline="") if analyze_result else None
    )

    try:
        for logical_error_model in logical_error_models:
            for n_cols, n_rows in params["qubit_layout"]:
                for J, h, dt, n_trotter in params["tfim"]:
                    for placement in params["placement_methods"]:
                        for n_aods in params["n_aods"]:
                            for (
                                trivial_ret,
                                tmr_assignment_method,
                                skip_rus,
                                decompose_move,
                                parallel_execution,
                            ) in params["settings"]:
                                for trial in range(params["trials_per_config"]):
                                    rng = np.random.default_rng(42 + trial)
                                    config = {
                                        "n_aods": n_aods,
                                        "consider_skip_rus": skip_rus,
                                        "trivial_return": trivial_ret,
                                        "decompose_move": decompose_move,
                                        "tmr_assignment_method": tmr_assignment_method,
                                        "rng": rng,
                                        "save_log": False,
                                    }
                                    print(
                                        "Compile setting:",
                                        config,
                                    )
                                    (
                                        qc_one_layer,
                                        rz_logs,
                                        profiling_results_per_case,
                                    ) = generate_one_layer_2d_tfim_circuit_star(
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

                                    if analyze_result:
                                        if (
                                            profiling_writer is None
                                            and profiling_results_per_case
                                        ):
                                            if profiling_file is None:
                                                raise RuntimeError(
                                                    "profiling_file is None while analyze_result is True"
                                                )
                                            profiling_writer = csv.DictWriter(
                                                profiling_file,
                                                fieldnames=profiling_results_per_case[
                                                    0
                                                ].keys(),
                                            )
                                            if profiling_header_needed:
                                                profiling_writer.writeheader()
                                        if profiling_writer is not None:
                                            for row in profiling_results_per_case:
                                                profiling_writer.writerow(row)

                                    result = simluate_trotter_2d_tfim_fidelity_star(
                                        n_qubits=n_cols * n_rows,
                                        n_factories=n_cols * n_rows,
                                        qubit_layout=(n_rows, n_cols),
                                        n_trotter_steps=1,  # for simplicity, we only evaluate 1 trotter step for star compilation
                                        qc_one_layer=qc_one_layer,
                                        execution_logs=rz_logs,
                                        logical_error_model=logical_error_model,
                                    )
                                    total_depth = None
                                    if analyze_result and profiling_results_per_case:
                                        total_depth = float(
                                            sum(
                                                float(r.get("total_time", 0.0))
                                                for r in profiling_results_per_case
                                            )
                                        )
                                    n_cnot = float(result.get("n_cnot", 0.0))
                                    n_h = float(result.get("n_h", 0.0))
                                    n_s = float(result.get("n_s", 0.0))
                                    p_i = float(
                                        logical_error_model.get_logical_error_rate("I")
                                    )
                                    if total_depth is None:
                                        n_idle = None
                                        fidelity_idle = None
                                        fidelity_with_idle = None
                                    else:
                                        n_idle = (
                                            (n_cols * n_rows) * total_depth
                                            - 2.0 * n_cnot
                                            - n_h
                                            - n_s
                                        )
                                        n_idle = max(0.0, float(n_idle))
                                        fidelity_idle = float((1.0 - p_i) ** n_idle)
                                        fidelity_with_idle = (
                                            float(result["fidelity"]) * fidelity_idle
                                        )
                                    row = {
                                        "trial": trial,
                                        "code_distance": logical_error_model.code_distance,
                                        "qubit_layout": (n_rows, n_cols),
                                        "n_trotter_steps": 1,
                                        "placement": placement,
                                        "n_aods": n_aods,
                                        "tmr_assignment_method": tmr_assignment_method,
                                        "consider_skip_rus": skip_rus,
                                        "trivial_return": trivial_ret,
                                        "decompose_move": decompose_move,
                                        "parallel_execution": parallel_execution,
                                        "total_depth": total_depth,
                                        "n_idle": n_idle,
                                        "fidelity_idle_model": fidelity_idle,
                                        "fidelity_with_idle": fidelity_with_idle,
                                        **result,
                                    }
                                    if result_writer is None:
                                        result_writer = csv.DictWriter(
                                            result_file,
                                            fieldnames=row.keys(),
                                        )
                                        if result_header_needed:
                                            result_writer.writeheader()
                                    result_writer.writerow(row)
    finally:
        result_file.close()
        if profiling_file is not None:
            profiling_file.close()


if __name__ == "__main__":
    # Define parameter grid
    qubit_layout = [
        (4, 4),  # 16 qubits
        # (5, 5),  # 25 qubits
        (6, 6),  # 36 qubits
        # (7, 7),  # 49 qubits
        (8, 8),  # 64 qubits
        # (9, 9),  # 81 qubits
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

    # fidelity evaluation for star compilation
    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": [2, 3, 4],
        "settings": [SETTINGS[4]],
        "trials_per_config": 5,
    }
    # run_evaluation_raw(params=params, physical_error_model=physical_error_model)

    logical_error_models = [
        LogicalErrorModel(physical_model=physical_error_model, code_distance=7),
        LogicalErrorModel(physical_model=physical_error_model, code_distance=9),
        LogicalErrorModel(physical_model=physical_error_model, code_distance=13),
    ]

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )

    # architecture study
    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["seperate_region_row", "checkerboard"],
        "n_aods": [1, 5],
        "settings": [SETTINGS[4]],
        "trials_per_config": 5,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )

    # ablation study
    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": [1, 5],
        "settings": SETTINGS,
        "trials_per_config": 5,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )
