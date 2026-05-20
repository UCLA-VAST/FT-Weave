from src.fidelity_simulation import (
    simluate_trotter_2d_tfim_fidelity,
    simluate_trotter_2d_tfim_fidelity_star,
)
from src.error_model import PhysicalErrorModel, LogicalErrorModel
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz
import os
from src.star.config import (
    get_config as get_star_config,
    update_config as update_star_config,
)
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
    """Run comprehensive evaluation with different parameter combinations.

    ``params`` may contain ``logical_se_interval`` to drive the logical-qubit
    SE scheduler. With it set, the per-layer execution log carries ``SE_q``
    events and the simulator returns a non-trivial ``fidelity_idle`` factor
    folded into ``fidelity``.
    """

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
    print(f"  logical_se_interval: {params.get('logical_se_interval')}")
    print(f"  prepare_lookahead_angles: {params.get('prepare_lookahead_angles')}")
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

    # Configure the logical-qubit SE scheduler globally for the duration of
    # this run; restore it afterwards so we don't leak settings into other
    # callers. ``get_star_config()`` returns derived read-only keys (e.g.
    # ``TMR_PREPARATION_TIME``) that ``update_star_config`` rejects, so we
    # only round-trip ``LOGICAL_SE_INTERVAL``.
    logical_se_interval = params.get("logical_se_interval")
    original_logical_se_interval = get_star_config().get("LOGICAL_SE_INTERVAL")
    update_star_config(LOGICAL_SE_INTERVAL=logical_se_interval)

    try:
        for logical_error_model in logical_error_models:
            for n_cols, n_rows in params["qubit_layout"]:
                for J, h, dt, n_trotter in params["tfim"]:
                    for placement in params["placement_methods"]:
                        for n_aods in params["n_aods"]:
                            for prepare_lookahead_angles in params.get(
                                "prepare_lookahead_angles", [True]
                            ):
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
                                            "prepare_lookahead_angles": prepare_lookahead_angles,
                                            "rng": rng,
                                            "save_log": False,
                                        }
                                        print(
                                            "Compile setting:",
                                            config,
                                        )
                                        (
                                            qc_one_layer,
                                            full_logs,
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
                                            execution_logs=full_logs,
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
                                        # ``result["fidelity"]`` already folds in
                                        # ``fidelity_idle`` (computed from the
                                        # SE_q events emitted by the logical-SE
                                        # scheduler). All counts and component
                                        # fidelities live in ``result``; we just
                                        # add per-trial bookkeeping here.
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
                                            "prepare_lookahead_angles": prepare_lookahead_angles,
                                            "logical_se_interval": logical_se_interval,
                                            "total_depth": total_depth,
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
        update_star_config(LOGICAL_SE_INTERVAL=original_logical_se_interval)
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
    # star_params = {
    #     "qubit_layout": qubit_layout,
    #     "tfim": tfim,
    #     "placement_methods": ["col_based"],
    #     "n_aods": [2, 3, 4],
    #     "settings": [SETTINGS[4]],
    #     "trials_per_config": 5,
    #     # Logical-qubit SE cadence (in cycles). ``None`` disables the
    #     # scheduler and ``fidelity_idle`` becomes 1.0.
    #     "logical_se_interval": 10,
    # }
    # run_evaluation_raw(params=params, physical_error_model=physical_error_model)

    logical_error_models = [
        # LogicalErrorModel(physical_model=physical_error_model, code_distance=7),
        LogicalErrorModel(physical_model=physical_error_model, code_distance=9),
        # LogicalErrorModel(physical_model=physical_error_model, code_distance=13),
    ]

    # run_evaluation_star(
    #     params=star_params,
    #     logical_error_models=logical_error_models,
    #     analyze_result=True,
    # )

    # # architecture study
    # star_params = {
    #     "qubit_layout": qubit_layout,
    #     "tfim": tfim,
    #     "placement_methods": ["seperate_region_row", "checkerboard"],
    #     "n_aods": [1, 5],
    #     "settings": [SETTINGS[4]],
    #     "trials_per_config": 5,
    #     "logical_se_interval": 10,
    # }

    # run_evaluation_star(
    #     params=star_params,
    #     logical_error_models=logical_error_models,
    #     analyze_result=True,
    # )

    # # ablation study
    # star_params = {
    #     "qubit_layout": qubit_layout,
    #     "tfim": tfim,
    #     "placement_methods": ["col_based"],
    #     "n_aods": [1, 5],
    #     "settings": SETTINGS,
    #     "trials_per_config": 5,
    #     "logical_se_interval": 10,
    # }

    # run_evaluation_star(
    #     params=star_params,
    #     logical_error_models=logical_error_models,
    #     analyze_result=True,
    # )

    # lookahead angle preparation ablation (current level only vs default)
    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": [2, 3, 4],
        "settings": [SETTINGS[4]],
        "prepare_lookahead_angles": [True, False],
        "trials_per_config": 5,
        "logical_se_interval": 10,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )

    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["seperate_region_row"],
        "n_aods": [1, 5],
        "settings": [SETTINGS[0]],
        "trials_per_config": 5,
        "logical_se_interval": 10,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )

    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": [1, 5],
        "settings": SETTINGS,
        "trials_per_config": 5,
        "logical_se_interval": 10,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )

    star_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": [2, 3, 4],
        "settings": [SETTINGS[4]],
        "trials_per_config": 5,
        "logical_se_interval": 10,
    }

    run_evaluation_star(
        params=star_params,
        logical_error_models=logical_error_models,
        analyze_result=True,
    )
