import os
import sys
from itertools import product
import csv
from datetime import datetime

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analog_rotation_execution import factory_angle_execution, get_microarchitecture

from src.ds import FactoryPool

from src.util import analyze_execution_log
from src.animator import (
    plot_all_rus_rounds,
    plot_circuit_execution,
    plot_circuit_execution_vertical,
)


def run_evaluation(params: dict, plot_figure: bool = False):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_configs = (
        len(params["qubit_sizes"])
        * len(params["placement_methods"])
        * len(params["n_aods"])
        * len(params["consider_skip_rus"])
        * len(params["tmr_assignment_method"])
        * len(params["trivial_return"])
    )

    print("=" * 80)
    print("EVALUATION SCRIPT - MAGIC STATE FACTORY EXECUTION")
    print("=" * 80)
    print(f"Total configurations to test: {total_configs}")
    print("=" * 80 + "\n")

    config_count = 0

    # Generate all parameter combinations
    for (
        (n_cols, n_rows),
        placement,
        n_aods,
        skip_rus,
        tmr_method,
        trivial_ret,
    ) in product(
        params["qubit_sizes"],
        params["placement_methods"],
        params["n_aods"],
        params["consider_skip_rus"],
        params["tmr_assignment_method"],
        params["trivial_return"],
    ):
        config_count += 1
        n_qubits = n_cols * n_rows
        n_factories = n_qubits

        target_qubits_angles = {}
        angle = 0.001
        for i in range(n_qubits):
            target_qubits_angles[i] = angle

        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories,
            (n_cols, n_rows),
            placement,
        )
        factory_pool = FactoryPool(num_factories=n_factories)
        config = {
            "target_qubits_angles": target_qubits_angles,
            "factory_pool": factory_pool,
            "logic_qubit_locations": logic_qubit_locations,
            "magic_state_locations": magic_state_locations,
            "n_aods": n_aods,
            "consider_skip_rus": skip_rus,
            "tmr_assignment_method": tmr_method,
            "trivial_return": trivial_ret,
        }

        print(f"[{config_count}/{total_configs}] Running configuration:")
        print(f"  Qubits: {n_qubits} ({n_cols}x{n_rows})")
        print(f"  Placement: {placement}")
        print(f"  AODs: {n_aods}")
        print(f"  Skip RUS: {skip_rus}")
        print(f"  TMR Method: {tmr_method}")
        print(f"  Trivial Return: {trivial_ret}")

        total_time, log = factory_angle_execution(
            **config,
        )
        profiling_result = analyze_execution_log(log, n_factories=n_factories)
        # print(profiling_result)
        result = {
            "config_id": config_count,
            # "status": "completed",
            # "timestamp": datetime.now().isoformat(),
            "n_qubits": n_qubits,
            "qubit_cols": n_cols,
            "qubit_rows": n_rows,
            "placement": placement,
            "n_aods": n_aods,
            "consider_skip_rus": skip_rus,
            "tmr_assignment_method": tmr_method,
            "trivial_return": trivial_ret,
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
        results.append(result)

        if plot_figure:
            base_path = f"_n{n_qubits}f{n_factories}_{placement}_naod_{n_aods}_tmr_{tmr_method}_skipRUS_{skip_rus}_trivial-return{trivial_ret}.pdf"
            pdf_path = f"output/evaluation/circuit_execution_vertical/{base_path}"
            plot_circuit_execution_vertical(
                log, n_factories, figure_height=50, save_path=pdf_path
            )
            pdf_path = f"output/evaluation/circuit_execution/{base_path}"
            plot_circuit_execution(
                log, n_factories, figure_width=50, save_path=pdf_path
            )
            pdf_path = f"output/evaluation/rus_rounds_detailed/{base_path}"
            plot_all_rus_rounds(
                execution_log=log,
                logic_qubit_locations=logic_qubit_locations,
                magic_state_locations=magic_state_locations,
                base_path=pdf_path,
            )

    # Save results to CSV
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    if results:
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Completed: {completed}/{total_configs}")
    print(f"Failed: {failed}/{total_configs}")
    print(f"Results saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    # Define parameter grid
    params = {
        "qubit_sizes": [
            # (4, 4),  # 16 qubits
            (5, 5),  # 25 qubits
            # (6, 6),  # 36 qubits
            # (7, 7),  # 49 qubits
            # (8, 8),  # 64 qubits
            # (9, 9),  # 81 qubits
            # (10, 10),  # 100 qubits
        ],
        "placement_methods": [
            "seperate_region_col",
            "col_based",
            # "checkerboard",
        ],
        # "n_aods": [1, 2, 3, 4, 5],
        "n_aods": [1, 2],
        # "consider_skip_rus": [False, True],
        "consider_skip_rus": [False, True],
        "tmr_assignment_method": ["naive", "matching"],
        "trivial_return": [False, True],
    }
    run_evaluation(params=params, plot_figure=True)
