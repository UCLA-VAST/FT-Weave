from script.raw_fidelity_simulation import simluate_trotter_2d_tfim_fidelity
from src.error_model import PhysicalErrorModel
from src.tfim_raw import generate_one_layer_2d_tfim_circuit_cz
import json
import os


def run_evaluation_raw(params: dict):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_configs = len(params["qubit_layout"]) * params["n_trotter_steps"]
    print("=" * 80)
    print("EVALUATION SCRIPT - RAW FIDELITY SIMULATION")
    print("=" * 80)
    print(f"Total configurations to test: {total_configs}")
    print("=" * 80 + "\n")
    # simulate raw physical fidelity for different qubit layouts and trotter steps
    for qubit_layout in params["qubit_layout"]:
        n_qubits = qubit_layout[0] * qubit_layout[1]
        for J, h, dt in params["tfim"]:
            qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
                n_qubits=n_qubits, qubit_layout=qubit_layout, J=J, h=h, dt=dt
            )
            for n_trotter_steps in params["n_trotter_steps"]:
                fidelity_profile = simluate_trotter_2d_tfim_fidelity(
                    n_qubits=n_qubits,
                    qubit_layout=qubit_layout,
                    n_trotter_steps=n_trotter_steps,
                    qc_one_layer=qc_one_layer,
                    physical_error_model=PhysicalErrorModel(),
                )
                result = {
                    "qubit_layout": qubit_layout,
                    "n_trotter_steps": n_trotter_steps,
                    **fidelity_profile,
                }
                results.append(result)
                print(result)

    # simulate raw physical fidelity for different qubit layouts and trotter steps

    # Save results to csv file
    results_path = os.path.join(output_dir, "raw_fidelity_results.csv")
    with open(results_path, "w") as f:
        # Write header
        f.write(
            "qubit_layout,n_trotter_steps,fidelity,fidelity_of_rz_layer,fidelity_cz,fidelity_1q,n_cz,n_h\n"
        )
        for result in results:
            f.write(
                f"{result['qubit_layout']},{result['n_trotter_steps']},{result['fidelity']},{result['fidelity_of_rz_layer']},{result['fidelity_cz']},{result['fidelity_1q']},{result['n_cz']},{result['n_h']}\n"
            )


def run_evaluation_star(params: dict):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_configs = (
        len(params["qubit_layout"])
        * len(params["n_trotter_steps"])
        * len(params["placement_methods"])
        * len(params["n_aods"])
        * len(params["consider_skip_rus"])
        * len(params["tmr_assignment_method"])
        * len(params["trivial_return"])
        * params["trials_per_config"]
        * len(params["decompose_move"])
        * len(params["parallel_execution"])
    )
    print("=" * 80)
    print("EVALUATION SCRIPT - STAR FIDELITY SIMULATION")
    print("=" * 80)
    print(f"Total configurations to test: {total_configs}")
    print("=" * 80 + "\n")
    raise NotImplementedError("Star fidelity simulation is not implemented yet.")


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

    tfim = [(1.0, 1.0, 0.1)]  # J, h, dt
    params = {
        "qubit_layout": qubit_layout,
        "n_trotter_steps": [1, 2, 3, 4, 5],
        "tfim": tfim,
    }
    star_params = {
        "qubit_layout": qubit_layout,
        "n_trotter_steps": [1, 2, 3, 4, 5],
        "tfim": tfim,
        "placement_methods": [
            "col_based",
            "checkerboard",
        ],
        "n_aods": [1, 2, 3, 4, 5],
        "consider_skip_rus": [0, 1, 2],
        "tmr_assignment_method": ["matching"],
        # "trivial_return": [False, True],
        "trivial_return": [False, True],
        "trials_per_config": 10,
        "decompose_move": [False],
        "parallel_execution": [False, True],
    }
    run_evaluation_raw(params=params)
