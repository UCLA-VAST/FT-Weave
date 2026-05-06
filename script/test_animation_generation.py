import json
import os
import sys
from copy import deepcopy

import numpy as np

# Ensure repository root is on sys.path so `src` and `script` are importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator.animator_matplotlib import Animator
from src.ds import Architecture, get_microarchitecture
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
from src.writer.execution_log_to_zair import execution_log_to_animator_code


def merge_layer_logs_with_global_timeline(
    layer_logs: list[list[dict]],
    inter_layer_gap: float = 0.0,
) -> list[dict]:
    """Concatenate per-layer logs into one monotonically increasing timeline."""
    merged: list[dict] = []
    current_time = 0.0
    for layer in layer_logs:
        if not layer:
            continue
        layer_start = min(float(entry.get("start_time", 0.0)) for entry in layer)
        layer_end = max(
            float(entry.get("end_time", entry.get("start_time", 0.0))) for entry in layer
        )
        shift = current_time - layer_start
        for entry in layer:
            new_entry = deepcopy(entry)
            start = float(new_entry.get("start_time", 0.0)) + shift
            end = float(new_entry.get("end_time", start)) + shift
            new_entry["start_time"] = start
            new_entry["end_time"] = end
            merged.append(new_entry)
        current_time += (layer_end - layer_start) + inter_layer_gap
    return merged


def main() -> None:
    # Fixed parameters for quick local testing (edit these directly as needed).
    n_cols = 2
    n_rows = 2
    placement = "col_based"
    code_distance = 3
    n_aods = 1
    parallel_execution = False
    seed = 0
    J = 1.0
    h = 0.7
    dt = 0.1
    max_layers = 12
    inter_layer_gap = 0.0
    arch_spec_path = "hardware_spec/logical_architecture.json"
    output_path = "output/animation/tfim_log_adapter_test.mp4"
    ffmpeg_path = "ffmpeg"

    n_qubits = n_cols * n_rows
    qubit_layout = (n_cols, n_rows)

    config = {
        "n_aods": n_aods,
        "consider_skip_rus": 0,
        "tmr_assignment_method": "matching",
        "trivial_return": False,
        "decompose_move": False,
        "rng": np.random.default_rng(seed),
    }

    _, layer_logs, _ = generate_one_layer_2d_tfim_circuit_star(
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        J=J,
        h=h,
        dt=dt,
        code_distance=code_distance,
        config=config,
        parallel_execution=parallel_execution,
        analyze_result=False,
    )

    layer_logs = layer_logs[:max_layers]
    merged_log = merge_layer_logs_with_global_timeline(
        layer_logs=layer_logs,
        inter_layer_gap=inter_layer_gap,
    )

    logic_qubit_locations, _ = get_microarchitecture(
        n_qubits=n_qubits,
        n_factories=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
    )

    with open(arch_spec_path, "r") as f:
        architecture = Architecture(json.load(f))

    code = execution_log_to_animator_code(
        merged_log,
        architecture=architecture,
        logic_qubit_locations=logic_qubit_locations,
        name="tfim-log-adapter-test",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Animator().animate(
        code=code,
        architecture=architecture,
        output=output_path,
        ffmpeg=ffmpeg_path,
    )
    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
